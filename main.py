#!/usr/bin/env python3
# coding : utf-8

import os
import sys
import datetime
import cv2
import dlib
import numpy as np
from optparse import OptionParser

# 画像と任意多角形の頂点群から多角形内部のHSV値を計算する
def calculate_average_hsv_in_polygon(image, polygon_points):
    # 多角形の輪郭点をNumPy配列に変換
    polygon_points = np.array(polygon_points, dtype=np.int32)

    # 多角形領域のマスクを作成
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_points], 255)

    # 画像をHSVに変換
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # マスクを適用して多角形領域のピクセルを抽出
    hsv_polygon_pixels = hsv_image[mask > 0]

    # 色相・彩度・明度の平均値を計算
    mean_hue = np.mean(hsv_polygon_pixels[:, 0])
    mean_saturation = np.mean(hsv_polygon_pixels[:, 1])
    mean_value = np.mean(hsv_polygon_pixels[:, 2])

    return mean_hue, mean_saturation, mean_value

# 画像Aの多角形内からHSVの値を計算し、画像BのHSVを画像Aに合わせる関数
def image_matching(image_a, image_b, polygon_points):
    # 画像AのHSV表現に変換
    hsv_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2HSV)
    h_a, s_a, v_a = cv2.split(hsv_a)

    # 画像BのHSV表現に変換
    hsv_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2HSV)
    h_b, s_b, v_b = cv2.split(hsv_b)

    # 画像Aの色相・彩度・明度の平均を求める
    mean_h_a, mean_s_a, mean_v_a = calculate_average_hsv_in_polygon(image_a, polygon_points)
    mean_h_b, mean_s_b, mean_v_b = calculate_average_hsv_in_polygon(image_b, polygon_points)

    # 画像Bの色相・彩度・明度を画像Aの平均に合わせる
    h_b_matched = h_b + (0 - mean_h_b)
    s_b_matched = s_b + (0 - mean_h_b)
    # v_b_matched = v_b + (0 - mean_h_b)
    # h_b_matched = h_b + (mean_h_a - mean_h_b)
    # s_b_matched = s_b + (mean_s_a - mean_h_b)
    v_b_matched = v_b + (mean_v_a - mean_h_b)

    # 合わせた色相・彩度・明度を結合して新しいHSV画像を作成
    hsv_b_matched = cv2.merge((h_b_matched, s_b_matched, v_b_matched))

    # 合わせたHSV画像の深度を適切な深度に変換
    hsv_b_matched = cv2.convertScaleAbs(hsv_b_matched, alpha=(255.0/360.0))

    # HSV画像をBGRに変換
    matched_image_b = cv2.cvtColor(hsv_b_matched, cv2.COLOR_HSV2BGR)

    return matched_image_b

def load_image(alpha_path, beta_path):
    # 画像の読み込み
    image_alpha = cv2.imread(alpha_path)
    image_beta = cv2.imread(beta_path)
    return image_alpha, image_beta

def get_face_landmarks(image):
    # Dlib の顔検出モデルを読み込み
    detector = dlib.get_frontal_face_detector()
    # 顔のランドマーク予測モデル
    predictor = dlib.shape_predictor("./facial-landmarks-recognition/shape_predictor_68_face_landmarks.dat")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return None, None

    landmarks = predictor(gray, faces[0])
    landmarks = np.array([(landmark.x, landmark.y) for landmark in landmarks.parts()])

    return faces[0], landmarks

def get_args():
    usage = "%prog image1 image2"
    parser = OptionParser(usage = usage)

    return parser.parse_args()

# alpha_path が使いたい顔・beta_path がハメたい身体
def collage(alpha_path, beta_path):
    # 画像の読み込み
    image_alpha, image_beta = load_image(alpha_path, beta_path)

    # ランドマークの取得
    face_alpha, landmarks_alpha = get_face_landmarks(image_alpha)
    face_beta, landmarks_beta = get_face_landmarks(image_beta)
    if face_alpha is None or face_beta is None:
        print(f"顔が検出できませんでした : alpha -> {not face_alpha is None} beta -> {not face_beta is None}")
        return 1, image_beta

    # 顔の領域を切り抜いて入れ替え
    src_pts = np.array([landmarks_alpha[0], landmarks_alpha[8], landmarks_alpha[16]], dtype = np.float32)
    dst_pts = np.array([landmarks_beta[0], landmarks_beta[8], landmarks_beta[16]], dtype = np.float32)
    # Affine 変換された alpha 画像を取得
    warped_face = cv2.warpAffine(image_alpha, cv2.getAffineTransform(src_pts, dst_pts), (image_beta.shape[1], image_beta.shape[0]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # 元の顔の周りを塗りつぶす
    # 黒地を作成、画像全体を黒くする
    mask = np.zeros_like(image_beta)
    # 白くする部分を決定、輪郭の点をプロット
    convexer = np.concatenate([landmarks_beta[0:17], landmarks_beta[26:16:-1]])
    # 輪郭の点の内側を白で塗りつぶし、黒地画像に上書き
    cv2.fillConvexPoly(mask, convexer, (255, 255, 255))
    # 輪郭の内側のHSVの平均値を計算し、warped_faceの画像のHSVを変更する
    warped_face = image_matching(image_beta, warped_face, convexer)
    # warped_face と mask で AND 計算。つまり warped_face から白い部分を抜き出す
    masked_face_alpha = cv2.bitwise_and(warped_face, mask)
    masked_body_beta = cv2.bitwise_and(image_beta, 255 - mask)

    # 入れ替えた顔を合成
    alpha_face_beta_body = cv2.add(masked_face_alpha, masked_body_beta)

    output = alpha_face_beta_body

    return 0, output

if __name__ == "__main__":
    optiondict, args = get_args()
    imageexies = [".jpg", ".jpeg", ".png", ".ping", ".webp", ".jfif", ".svg", ".pgm", ".jpg_large"]

    alpha_dir = ""
    beta_dir = ""

    if len(args) < 2 or len(args) > 2:
        print("画像を２つ指定しろ")
        sys.exit(1)
    else:
        alpha_dir = args[0]
        beta_dir = args[1]

    ld_alpha = os.listdir(alpha_dir)
    ld_beta = os.listdir(beta_dir)
    ld_alpha.sort()
    ld_beta.sort()

    now = datetime.datetime.now()
    output_dir = f"./output/{now.year:0>4}{now.month:0>2}{now.day:0>2}{now.hour:0>2}{now.minute:0>2}{now.second:0>2}"
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError:
            os.makedirs(output_dir[0:32])
    for ai, alpha_path in enumerate(ld_alpha):
        for bi, beta_path in enumerate(ld_beta):
            if (os.path.splitext(alpha_path)[-1].lower() in imageexies) and (os.path.splitext(beta_path)[-1].lower() in imageexies):
                print(f"cllage {alpha_path} and {beta_path}")
                res, image = collage(alpha_dir + "/" + alpha_path, beta_dir + "/" + beta_path)
                if res == 0:
                    fpath = f"{output_dir}/{ai}{bi}.png"
                    cv2.imwrite(fpath, image)
            else:
                print(f"skip {alpha_path.lower()} and {beta_path.lower()}")

