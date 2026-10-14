#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gaze_validation_live.py
即時比對:webcam CV 追瞳孔  vs  Tobii 眼動儀  vs  EOG (HEOG = HR - HL)

畫面
  左   : webcam 即時畫面(MediaPipe 虹膜/眼角標記,鏡像顯示,像照鏡子)
  右 1 : CV gaze-x     (0 = 受試者左, 1 = 受試者右)
  右 2 : Tobii gaze-x  (display area 正規化座標)
  右 3 : HEOG = HR(EXG5) - HL(EXG1) (uV),causal 帶通 0.05-20 Hz(右看為正)
  下   : 滑動視窗 Pearson 相關係數 r(CV,EOG) / r(Tobii,EOG) / r(CV,Tobii)

時間對齊
  三路資料全部換算到 LSL 時鐘 (pylsl.local_clock) 再存檔、畫圖、算相關。

安裝
  python -m pip install -r requirements.txt   (不需要 scipy;濾波用 numpy 自行實作)

用法
  python gaze_validation_live.py --simulate                    # 沒接硬體先試畫面
  python gaze_validation_live.py                               # BioSemi LSL,預設 HR=EXG5、HL=EXG1
  python gaze_validation_live.py --hr EXG5 --hl EXG1           # 通道標籤(或從 0 開始的編號)
  python gaze_validation_live.py --cv-delay 0.08               # 扣掉 webcam 固定延遲
按 Esc 或 Q 結束。
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
import threading
import time
from datetime import datetime

import numpy as np
import cv2
import pylsl
from pylsl import StreamInlet, local_clock, resolve_byprop

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets


# --------------------------------------------------------------------------
# 參數
# --------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CV vs Tobii vs EOG 即時比對")
    p.add_argument("--simulate", action="store_true", help="使用模擬訊號,不接任何硬體")
    # CV
    p.add_argument("--camera", type=int, default=0, help="webcam 編號")
    p.add_argument("--cv-delay", type=float, default=0.0,
                   help="webcam 固定延遲(秒),會從 CV 時間戳中扣掉")
    p.add_argument("--cv-gain", type=float, default=1.0,
                   help="以 0.5 為中心放大 CV gaze-x,只影響顯示尺度,不影響相關係數")
    p.add_argument("--face-model", default=None,
                   help="face_landmarker.task 路徑(新版 mediapipe 才需要;預設放在程式旁邊,沒有就自動下載)")
    p.add_argument("--invert-cv", action="store_true",
                   help="若往右看時 CV 曲線反而下降(攝影機被驅動程式鏡像),加上此旗標")
    # Tobii
    p.add_argument("--no-tobii", action="store_true")
    p.add_argument("--tobii-either-eye", action="store_true",
                   help="任一眼有效就計算 gaze-x(預設:左右眼都有效才計算,與離線分析一致)")
    p.add_argument("--tobii-address", default=None, help="例如 tobii-ttp://TPNA1-xxxx;不填則取第一台")
    # EOG
    p.add_argument("--no-eog", action="store_true")
    p.add_argument("--eeg-stream", default="BioSemi", help="BioSemi LSL 串流名稱")
    p.add_argument("--hr", default="EXG5", help="右眼外眥電極 HR 的通道標籤(或從 0 開始的編號)")
    p.add_argument("--hl", default="EXG1", help="左眼外眥電極 HL 的通道標籤(或從 0 開始的編號)")
    p.add_argument("--bp", type=float, nargs=2, default=[0.05, 20.0], metavar=("LOW", "HIGH"))
    # 顯示與相關
    p.add_argument("--window", type=float, default=10.0, help="畫面滾動視窗(秒)")
    p.add_argument("--corr-window", type=float, default=5.0, help="相關係數滑動視窗(秒)")
    p.add_argument("--corr-interval", type=float, default=0.5, help="相關係數更新間隔(秒)")
    # 輸出
    p.add_argument("--out", default="recordings")
    p.add_argument("--no-record", action="store_true")
    p.add_argument("--duration", type=float, default=0.0, help="自動結束秒數,0 = 手動結束")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# 共用小工具:緩衝區、CSV
# --------------------------------------------------------------------------
class SignalBuffer:
    """執行緒安全的 (時間, 數值) 緩衝區。"""

    def __init__(self, maxlen: int):
        self.t = collections.deque(maxlen=maxlen)
        self.v = collections.deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def append(self, t, v):
        with self.lock:
            self.t.append(t)
            self.v.append(v)

    def extend(self, ts, vs):
        with self.lock:
            self.t.extend(ts)
            self.v.extend(vs)

    def snapshot(self, t_min: float):
        with self.lock:
            t = np.array(self.t, dtype=float)
            v = np.array(self.v, dtype=float)
        m = t >= t_min
        return t[m], v[m]


class CsvLogger:
    def __init__(self, path, header):
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(header)
        self.lock = threading.Lock()
        self._last_flush = time.time()

    def _maybe_flush(self):
        if time.time() - self._last_flush > 1.0:
            self.f.flush()
            self._last_flush = time.time()

    def write(self, row):
        with self.lock:
            self.w.writerow(row)
            self._maybe_flush()

    def writemany(self, arr):
        with self.lock:
            self.w.writerows(np.asarray(arr).tolist())
            self._maybe_flush()

    def close(self):
        with self.lock:
            self.f.close()


class NullLogger:
    def write(self, row): pass
    def writemany(self, arr): pass
    def close(self): pass


# --------------------------------------------------------------------------
# 三路資料的「接收端」:負責放進緩衝區並寫 CSV
# --------------------------------------------------------------------------
class CVSink:
    def __init__(self, buf, logger):
        self.buf, self.logger = buf, logger

    def push(self, t, idx, gx, r_right, r_left):
        self.buf.append(t, gx)
        self.logger.write([t, idx, gx, r_right, r_left])


class TobiiSink:
    def __init__(self, buf, logger):
        self.buf, self.logger = buf, logger

    def push(self, t, dev_ts, lx, ly, lv, rx, ry, rv, gx):
        self.buf.append(t, gx)
        self.logger.write([t, dev_ts, lx, ly, lv, rx, ry, rv, gx])


class Biquad:
    """二階 Butterworth(雙線性轉換)濾波區段,直接 II 轉置型,逐 chunk 保留狀態。
    只用 numpy/內建,不需要 scipy(你的環境 scipy.signal 無法匯入)。"""

    def __init__(self, kind: str, fc: float, fs: float):
        K = np.tan(np.pi * fc / fs)
        norm = 1.0 / (1.0 + np.sqrt(2.0) * K + K * K)
        self.a1 = 2.0 * (K * K - 1.0) * norm
        self.a2 = (1.0 - np.sqrt(2.0) * K + K * K) * norm
        if kind == "lowpass":
            b0 = K * K * norm
            self.b = (b0, 2.0 * b0, b0)
        elif kind == "highpass":
            self.b = (norm, -2.0 * norm, norm)
        else:
            raise ValueError(kind)
        self.z1 = self.z2 = 0.0

    def init_steady(self, x0: float):
        """假設輸入一直是常數 x0,把狀態設成穩態(避免啟動暫態)。"""
        b0, b1, b2 = self.b
        y_ss = (b0 + b1 + b2) / (1.0 + self.a1 + self.a2) * x0
        self.z1 = y_ss - b0 * x0
        self.z2 = b2 * x0 - self.a2 * y_ss

    def process(self, x):
        b0, b1, b2 = self.b
        a1, a2 = self.a1, self.a2
        z1, z2 = self.z1, self.z2
        out = [0.0] * len(x)
        for i, xi in enumerate(np.asarray(x, dtype=float).tolist()):
            y = b0 * xi + z1
            z1 = b1 * xi - a1 * y + z2
            z2 = b2 * xi - a2 * y
            out[i] = y
        self.z1, self.z2 = z1, z2
        return np.asarray(out)


class EOGSink:
    """HEOG = HR - HL,再做 causal 帶通(高通 + 低通各一個二階 Butterworth,逐 chunk 連續濾波)。"""

    SAT_LIMIT = 199999.0  # BioSemi 電極脫落/飽和時輸出約 ±200000,與離線 load_eeg 的判定一致

    def __init__(self, buf, logger, band):
        self.buf, self.logger, self.band = buf, logger, band
        self.hp = self.lp = None
        self.started = False
        self.last_good = 0.0

    def configure(self, fs: float):
        low, high = self.band
        self.hp = Biquad("highpass", low, fs)
        self.lp = Biquad("lowpass", high, fs)
        self.started = False
        self.last_good = 0.0

    def push(self, ts, hr, hl):
        raw = hr - hl
        sat = (np.abs(hr) >= self.SAT_LIMIT) | (np.abs(hl) >= self.SAT_LIMIT)
        # 飽和樣本:送進濾波器時沿用前一個有效值,避免極端值汙染 IIR 狀態;
        # 輸出(畫面、相關、CSV 的 heog_filt)則把這些樣本標成 NaN
        raw_f = raw.copy()
        if sat.any():
            for k in range(len(raw_f)):
                if sat[k]:
                    raw_f[k] = self.last_good
                else:
                    self.last_good = raw_f[k]
        elif len(raw_f):
            self.last_good = raw_f[-1]
        if not self.started:  # 以第一個樣本當作 DC,避免啟動時的大暫態
            self.hp.init_steady(raw_f[0])
            self.lp.init_steady(0.0)  # 高通輸出的穩態為 0
            self.started = True
        y = self.lp.process(self.hp.process(raw_f))
        y[sat] = np.nan
        self.buf.extend(ts, y)
        self.logger.writemany(np.column_stack([ts, hr, hl, raw, y]))


# --------------------------------------------------------------------------
# CV:webcam + MediaPipe 虹膜
# --------------------------------------------------------------------------
def eye_ratio(lm, corner_a, corner_b, iris, invert=False):
    """
    回傳該眼的水平視線比例(受試者視角:0 = 左, 1 = 右)。
    webcam 未鏡像時,受試者往右看 -> 虹膜在影像中往左移,所以用 1 - 比例。
    """
    xa, xb = lm[corner_a].x, lm[corner_b].x
    lo, hi = min(xa, xb), max(xa, xb)
    if hi - lo < 1e-4:
        return np.nan
    r = 1.0 - (lm[iris].x - lo) / (hi - lo)
    return 1.0 - r if invert else r


FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")


def ensure_face_model(path=None):
    """回傳 face_landmarker.task 的路徑;不存在就下載到程式同一個資料夾。"""
    import urllib.request
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
    if not os.path.exists(path):
        print(f"[CV] 下載臉部地標模型到 {path} ...")
        try:
            urllib.request.urlretrieve(FACE_MODEL_URL, path)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"模型下載失敗({e})。請手動下載 {FACE_MODEL_URL} 並用 --face-model 指定路徑")
    return path


class CVThread(threading.Thread):
    # MediaPipe FaceMesh 地標:右眼(影像左側) 眼角 33/133、虹膜中心 468;左眼 眼角 362/263、虹膜中心 473
    R_EYE = (33, 133, 468)
    L_EYE = (362, 263, 473)

    def __init__(self, args, sink):
        super().__init__(daemon=True)
        self.args, self.sink = args, sink
        self.stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._frame, self._frame_id = None, 0

    def get_frame(self):
        with self._lock:
            return self._frame_id, self._frame

    def _make_detector(self):
        """
        回傳 detect(rgb) -> 地標 list 或 None。
        舊版 mediapipe 有 mp.solutions,直接用;新版(0.10.30 起)已移除,改用 Tasks API 的 FaceLandmarker
        (需要 face_landmarker.task 模型檔,找不到時自動下載)。兩者的 478 點地標編號相同。
        """
        import mediapipe as mp
        if hasattr(mp, "solutions"):
            mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)

            def detect(rgb):
                res = mesh.process(rgb)
                return res.multi_face_landmarks[0].landmark if res.multi_face_landmarks else None
            return detect

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        model = ensure_face_model(self.args.face_model)
        landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model),
            running_mode=vision.RunningMode.VIDEO, num_faces=1))
        last_ts = [0]

        def detect(rgb):
            ts = max(int(time.monotonic() * 1000), last_ts[0] + 1)  # VIDEO 模式要求時間戳嚴格遞增
            last_ts[0] = ts
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            res = landmarker.detect_for_video(img, ts)
            return res.face_landmarks[0] if res.face_landmarks else None
        return detect

    def run(self):
        try:
            detect = self._make_detector()
        except ImportError:
            print("[CV] 找不到 mediapipe,請先 pip install mediapipe")
            return
        except Exception as e:  # noqa: BLE001
            print(f"[CV] 無法初始化臉部地標偵測:{e}")
            return
        cap = cv2.VideoCapture(self.args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            print(f"[CV] 打不開 camera {self.args.camera}")
            return
        print("[CV] webcam 已啟動")
        idx = 0
        while not self.stop_evt.is_set():
            ok, frame = cap.read()
            t = local_clock() - self.args.cv_delay
            if not ok:
                continue
            h, w = frame.shape[:2]
            lm = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            gx = rr = lr = np.nan
            if lm is not None:
                rr = eye_ratio(lm, *self.R_EYE, invert=self.args.invert_cv)
                lr = eye_ratio(lm, *self.L_EYE, invert=self.args.invert_cv)
                vals = [v for v in (rr, lr) if np.isfinite(v)]
                if vals:
                    gx = 0.5 + self.args.cv_gain * (float(np.mean(vals)) - 0.5)
                for i in (*self.R_EYE, *self.L_EYE):
                    cv2.circle(frame, (int(lm[i].x * w), int(lm[i].y * h)), 2, (0, 255, 255), -1)
            self.sink.push(t, idx, gx, rr, lr)
            disp = cv2.flip(frame, 1)  # 鏡像顯示
            txt = f"CV gaze x = {gx:.2f}" if np.isfinite(gx) else "no face"
            cv2.putText(disp, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._frame, self._frame_id = disp, self._frame_id + 1
            idx += 1
        cap.release()

    def stop(self):
        self.stop_evt.set()


# --------------------------------------------------------------------------
# Tobii:tobii_research 串流
# --------------------------------------------------------------------------
class TobiiSource:
    def __init__(self, args, sink):
        self.args, self.sink = args, sink
        self.tr, self.tracker = None, None
        self._offsets = collections.deque(maxlen=600)

    def start(self):
        try:
            import tobii_research as tr
        except ImportError:
            print("[Tobii] 找不到 tobii_research,略過眼動儀")
            return
        if self.args.tobii_address:
            tracker = tr.EyeTracker(self.args.tobii_address)
        else:
            found = tr.find_all_eyetrackers()
            if not found:
                print("[Tobii] 找不到眼動儀,略過")
                return
            tracker = found[0]
        self.tr, self.tracker = tr, tracker
        print(f"[Tobii] 使用 {tracker.model} / {tracker.serial_number}(請先完成校正)")
        tracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, self._cb, as_dictionary=True)

    def _cb(self, g):
        now = local_clock()
        sys_t = g["system_time_stamp"] / 1e6
        # 回呼一定有傳遞延遲(>=0),取近 10 秒最小的 (LSL - system) 差值當作時鐘偏移
        self._offsets.append(now - sys_t)
        t = sys_t + min(self._offsets)
        xs, vals = [], {}
        for side in ("left", "right"):
            x, y = g[f"{side}_gaze_point_on_display_area"]
            v = int(g[f"{side}_gaze_point_validity"])
            vals[side] = (x, y, v)
            if v:
                xs.append(x)
        # 與離線 detect_saccades 相同:預設要左右眼都有效才算 gaze-x;--tobii-either-eye 則任一眼有效即可
        need = 1 if self.args.tobii_either_eye else 2
        gx = float(np.mean(xs)) if len(xs) >= need else np.nan
        self.sink.push(t, g["device_time_stamp"], *vals["left"], *vals["right"], gx)

    def stop(self):
        if self.tracker is not None:
            try:
                self.tracker.unsubscribe_from(self.tr.EYETRACKER_GAZE_DATA, self._cb)
            except Exception:
                pass


# --------------------------------------------------------------------------
# EOG:BioSemi 經 LSL
# --------------------------------------------------------------------------
def get_labels(info):
    labels = []
    try:
        ch = info.desc().child("channels").child("channel")
        for _ in range(info.channel_count()):
            labels.append(ch.child_value("label"))
            ch = ch.next_sibling()
    except Exception:
        pass
    return labels


def resolve_channel(spec, labels, nch):
    try:
        i = int(spec)
    except ValueError:
        if spec not in labels:
            raise ValueError(f"串流中找不到通道標籤 '{spec}'")
        i = labels.index(spec)
    if not 0 <= i < nch:
        raise ValueError(f"通道編號 {i} 超出範圍 (0-{nch - 1})")
    return i


class EOGThread(threading.Thread):
    def __init__(self, args, sink):
        super().__init__(daemon=True)
        self.args, self.sink = args, sink
        self.stop_evt = threading.Event()

    def run(self):
        try:
            self._run()
        except Exception as e:  # noqa: BLE001
            print(f"[EOG] 發生錯誤:{e}")

    def _run(self):
        print(f"[EOG] 尋找 LSL 串流 '{self.args.eeg_stream}' ...")
        infos = resolve_byprop("name", self.args.eeg_stream, timeout=5)
        if not infos:
            infos = resolve_byprop("type", "EEG", timeout=3)
        if not infos:
            print("[EOG] 找不到 LSL 串流,請確認 BioSemi LSL 已開啟,或用 --eeg-stream 指定名稱")
            return
        inlet = StreamInlet(infos[0], max_buflen=30,
                            processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter | pylsl.proc_monotonize)
        info = inlet.info()
        nch, fs = info.channel_count(), info.nominal_srate()
        labels = get_labels(info)
        print(f"[EOG] 已連線:{info.name()}  {nch} ch @ {fs:g} Hz")
        if labels:
            print("[EOG] 通道:" + ", ".join(f"{i}:{l}" for i, l in enumerate(labels)))
        ir = resolve_channel(self.args.hr, labels, nch)
        il = resolve_channel(self.args.hl, labels, nch)
        print(f"[EOG] HEOG = ch{ir} - ch{il}")
        self.sink.configure(fs)
        while not self.stop_evt.is_set():
            samples, ts = inlet.pull_chunk(timeout=0.05, max_samples=4096)
            if not ts:
                continue
            arr = np.asarray(samples, dtype=float)
            self.sink.push(np.asarray(ts, dtype=float), arr[:, ir], arr[:, il])

    def stop(self):
        self.stop_evt.set()


# --------------------------------------------------------------------------
# 模擬器(沒接硬體時用):同一條「真實視線」產生三路帶雜訊/延遲的訊號
# --------------------------------------------------------------------------
class Simulator(threading.Thread):
    def __init__(self, cv_sink, tb_sink, eog_sink, fs_eog=512.0):
        super().__init__(daemon=True)
        self.cv, self.tb, self.eog, self.fs = cv_sink, tb_sink, eog_sink, fs_eog
        self.stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._frame, self._frame_id = None, 0

    def get_frame(self):
        with self._lock:
            return self._frame_id, self._frame

    def run(self):
        rng = np.random.default_rng(0)
        jt = np.cumsum(rng.uniform(0.3, 1.2, 5000))
        jv = np.concatenate([[0.5], rng.uniform(0.2, 0.8, 5000)])
        gaze = lambda tau: jv[np.searchsorted(jt, tau, side="right")]  # noqa: E731
        self.eog.configure(self.fs)
        t0 = local_clock()
        k, next_cv, next_tb, idx = 0, t0, t0, 0
        while not self.stop_evt.is_set():
            now = local_clock()
            n_new = int((now - t0) * self.fs) - k
            if n_new > 0:
                ts = t0 + (k + np.arange(n_new)) / self.fs
                tau = ts - t0
                heog = 400 * (gaze(tau) - 0.5) + 60 * np.sin(2 * np.pi * 0.03 * tau) + rng.normal(0, 3, n_new)
                self.eog.push(ts, 0.5 * heog + 1000, -0.5 * heog + 1000)  # 帶 DC 偏移,測試濾波器
                k += n_new
            while next_cv <= now:  # CV:30 Hz,延遲 60 ms,振幅較小
                g = 0.5 + 0.25 * (gaze(next_cv - t0 - 0.06) - 0.5) + rng.normal(0, 0.01)
                self.cv.push(next_cv, idx, g, g, g)
                frame = np.zeros((480, 640, 3), np.uint8) + 40
                cv2.circle(frame, (int(g * 640), 240), 18, (0, 255, 255), -1)
                cv2.putText(frame, "SIMULATION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                with self._lock:
                    self._frame, self._frame_id = frame, self._frame_id + 1
                idx, next_cv = idx + 1, next_cv + 1 / 30
            while next_tb <= now:  # Tobii:60 Hz,延遲 10 ms
                g = float(np.clip(gaze(next_tb - t0 - 0.01) + rng.normal(0, 0.01), 0, 1))
                self.tb.push(next_tb, 0, g, 0.5, 1, g, 0.5, 1, g)
                next_tb += 1 / 60
            time.sleep(0.005)

    def stop(self):
        self.stop_evt.set()


# --------------------------------------------------------------------------
# 滑動視窗相關
# --------------------------------------------------------------------------
def resample_gappy(grid, t, v, max_gap=0.2):
    """線性內插到共同時間格;相鄰有效樣本間隔 > max_gap(如眨眼、追蹤遺失)的地方設為 NaN。"""
    m = np.isfinite(v)
    t, v = t[m], v[m]
    if len(t) < 5:
        return np.full_like(grid, np.nan)
    out = np.interp(grid, t, v, left=np.nan, right=np.nan)
    j = np.searchsorted(t, grid)
    left = t[np.clip(j - 1, 0, len(t) - 1)]
    right = t[np.clip(j, 0, len(t) - 1)]
    out[(right - left) > max_gap] = np.nan
    return out


def window_corr(a, b, t0, t1, fs=100.0, min_valid=0.4):
    """a, b = (t, v) 陣列。回傳 (Pearson r, 有效比例)。"""
    grid = np.arange(t0, t1, 1.0 / fs)
    ga, gb = resample_gappy(grid, *a), resample_gappy(grid, *b)
    m = np.isfinite(ga) & np.isfinite(gb)
    frac = m.mean() if len(grid) else 0.0
    if frac < min_valid or ga[m].std() < 1e-9 or gb[m].std() < 1e-9:
        return np.nan, frac
    return float(np.corrcoef(ga[m], gb[m])[0, 1]), frac


def fmt(r):
    return f"{r:+.2f}" if np.isfinite(r) else "  — "


# --------------------------------------------------------------------------
# 主視窗
# --------------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, args, frame_src, bufs, corr_logger, stop_fn):
        super().__init__()
        self.args, self.frame_src, self.bufs = args, frame_src, bufs
        self.corr_logger, self.stop_fn = corr_logger, stop_fn
        self._last_fid, self._last_corr = -1, 0.0
        self.setWindowTitle("CV vs Tobii vs EOG — live")
        self.resize(1600, 820)

        w = pg.GraphicsLayoutWidget()
        self.setCentralWidget(w)
        self.vb = w.addViewBox(row=0, col=0, rowspan=3, lockAspect=True, invertY=True)
        self.vb.setMouseEnabled(False, False)
        self.vb.setMenuEnabled(False)
        self.img = pg.ImageItem()
        self.vb.addItem(self.img)

        specs = [("CV pupil tracking (webcam)", "CV gaze x", (0, 0, 0)),
                 ("Tobii eye tracker", "Tobii gaze x", (0, 0, 0)),
                 ("EOG: HR - HL", "HEOG (uV)", (0, 114, 189))]
        self.plots, self.curves, self.titles = [], [], []
        for i, (title, ylab, color) in enumerate(specs):
            pl = w.addPlot(row=i, col=1)
            pl.showGrid(x=True, y=True, alpha=0.3)
            pl.setLabel("left", ylab)
            pl.getAxis("left").enableAutoSIPrefix(False)  # 避免座標軸出現 (x0.001)
            pl.setXRange(-args.window, 0, padding=0)
            pl.setTitle(title, size="10pt")
            if i < 2:
                pl.getAxis("bottom").setStyle(showValues=False)
            else:
                pl.setLabel("bottom", "time relative to now (s)")
            if i > 0:
                pl.setXLink(self.plots[0])
            self.plots.append(pl)
            self.titles.append(title)
            self.curves.append(pl.plot(pen=pg.mkPen(color, width=1.5)))
        self.plots[1].setYRange(0, 1)
        self.info = w.addLabel("", row=3, col=0, colspan=2, justify="left")
        w.ci.layout.setColumnStretchFactor(0, 2)
        w.ci.layout.setColumnStretchFactor(1, 3)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_view)
        self.timer.start(33)
        if args.duration > 0:
            QtCore.QTimer.singleShot(int(args.duration * 1000), self.close)

    def update_view(self):
        now = local_clock()
        t0 = now - self.args.window
        fid, frame = self.frame_src.get_frame() if self.frame_src else (-1, None)
        if frame is not None and fid != self._last_fid:
            self.img.setImage(frame, autoLevels=False, levels=(0, 255))
            self._last_fid = fid
        for key, curve in zip(("cv", "tobii", "eog"), self.curves):
            t, v = self.bufs[key].snapshot(t0)
            if len(t):
                curve.setData(t - now, v, connect="finite")
            else:
                curve.setData([], [])
        if now - self._last_corr >= self.args.corr_interval:
            self._last_corr = now
            self.update_corr(now)

    def update_corr(self, now):
        W = self.args.corr_window
        t0, t1 = now - W, now
        s = {k: self.bufs[k].snapshot(t0 - 0.5) for k in ("cv", "tobii", "eog")}
        r_ce, f_ce = window_corr(s["cv"], s["eog"], t0, t1)
        r_te, f_te = window_corr(s["tobii"], s["eog"], t0, t1)
        r_ct, f_ct = window_corr(s["cv"], s["tobii"], t0, t1)
        self.plots[0].setTitle(f"{self.titles[0]}   |   r vs EOG = {fmt(r_ce)}", size="10pt")
        self.plots[1].setTitle(f"{self.titles[1]}   |   r vs EOG = {fmt(r_te)}", size="10pt")
        self.info.setText(
            f"sliding window {W:g}s   |   r(CV, EOG) = {fmt(r_ce)}   r(Tobii, EOG) = {fmt(r_te)}"
            f"   r(CV, Tobii) = {fmt(r_ct)}   |   valid: {f_ce:.0%} / {f_te:.0%} / {f_ct:.0%}"
            f"   |   {'REC' if not self.args.no_record else 'not recording'}   (Esc / Q to quit)",
            color="k", size="11pt")
        self.corr_logger.write([now, r_ce, r_te, r_ct])

    def keyPressEvent(self, e):
        if e.key() in (QtCore.Qt.Key_Escape, QtCore.Qt.Key_Q):
            self.close()

    def closeEvent(self, e):
        self.timer.stop()
        self.stop_fn()
        super().closeEvent(e)


# --------------------------------------------------------------------------
# 組裝與啟動
# --------------------------------------------------------------------------
def build(args):
    pg.setConfigOptions(imageAxisOrder="row-major", background="w", foreground="k")

    run_dir = None
    if not args.no_record:
        run_dir = os.path.join(args.out, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "meta.json"), "w") as f:
            json.dump({"args": vars(args), "wallclock_at_start": time.time(),
                       "lsl_clock_at_start": local_clock()}, f, indent=2)
        print(f"[REC] 存檔資料夾:{run_dir}")

    def mk(name, header):
        return CsvLogger(os.path.join(run_dir, name), header) if run_dir else NullLogger()

    log_cv = mk("cv_gaze.csv", ["lsl_time", "frame", "gaze_x", "ratio_right_eye", "ratio_left_eye"])
    log_tb = mk("tobii_gaze.csv", ["lsl_time", "device_ts_us", "l_x", "l_y", "l_valid",
                                    "r_x", "r_y", "r_valid", "gaze_x"])
    log_eog = mk("eog.csv", ["lsl_time", "hr_raw", "hl_raw", "heog_raw", "heog_filt"])
    log_corr = mk("corr.csv", ["lsl_time", "r_cv_eog", "r_tobii_eog", "r_cv_tobii"])

    keep = max(args.window, args.corr_window) + 5
    bufs = {"cv": SignalBuffer(int(keep * 120)),
            "tobii": SignalBuffer(int(keep * 150)),
            "eog": SignalBuffer(int(keep * 4200))}
    cv_sink, tb_sink = CVSink(bufs["cv"], log_cv), TobiiSink(bufs["tobii"], log_tb)
    eog_sink = EOGSink(bufs["eog"], log_eog, args.bp)

    workers, tobii, frame_src = [], None, None
    if args.simulate:
        sim = Simulator(cv_sink, tb_sink, eog_sink)
        workers.append(sim)
        frame_src = sim
    else:
        cvt = CVThread(args, cv_sink)
        workers.append(cvt)
        frame_src = cvt
        if not args.no_eog:
            workers.append(EOGThread(args, eog_sink))
        if not args.no_tobii:
            tobii = TobiiSource(args, tb_sink)
            tobii.start()

    for wkr in workers:
        wkr.start()

    def stop():
        for wkr in workers:
            wkr.stop()
        if tobii:
            tobii.stop()
        time.sleep(0.2)
        for lg in (log_cv, log_tb, log_eog, log_corr):
            lg.close()

    app = pg.mkQApp("gaze-validation")
    win = MainWindow(args, frame_src, bufs, log_corr, stop)
    return app, win


def main():
    args = parse_args()
    app, win = build(args)
    win.show()
    sys.exit(app.exec() if hasattr(app, "exec") else app.exec_())


if __name__ == "__main__":
    main()
