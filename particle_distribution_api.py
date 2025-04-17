import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime
import tempfile
import base64
import traceback

app = Flask(__name__)
CORS(app)  # 啟用 CORS 支援跨域請求

# 設定儲存上傳檔案和生成結果的目錄
UPLOAD_FOLDER = tempfile.mkdtemp()
RESULTS_FOLDER = tempfile.mkdtemp()
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ---------------------------- 分析函式 ----------------------------
def analyze_partical_distribution(image_path, channel, output_dir=None):
    """
    分析器官晶片(OoC)流道內螢光粒子分布的影像檔案，並依據指定的 channel 處理圖像。

    參數:
        image_path (str): 影像檔案路徑，例如 "./500 ml_particle_RGB_TRITC.tif" 或其他格式
        channel (str): 指定通道 ("red" ch0 , "green" ch1 , "blue" ch2 ，不分大小寫)
        output_dir (str): 輸出目錄，預設為 None

    回傳:
        dict: 結果字典，包含分析結果和生成的檔案路徑
    """
    if output_dir is None:
        output_dir = os.path.dirname(image_path)

    #--------------------------------------------------------------
    # 1. 讀圖
    # --------------------------------------------------------------
    channel_map = {'red': 0, 'green': 1, 'blue': 2}
    if channel.lower() not in channel_map:
        raise ValueError("Channel 必須為 'red', 'green' 或 'blue'")
    channel_index = channel_map[channel.lower()]

    # 根據副檔名讀取影像：TIFF 使用 tifffile，其它格式用 PIL 讀取
    ext = os.path.splitext(image_path)[1].lower()
    try:
        if ext in ['.tif', '.tiff']:
            img = tifffile.imread(image_path)
        else:
            img = np.array(Image.open(image_path))
    except Exception as e:
        return {"error": f"無法讀取影像檔案：{str(e)}"}

    # 處理影像尺寸，支援 (H, W, 3) 或 (3, H, W) 格式
    if img.ndim == 3:
        if img.shape[0] == 3 and img.shape[-1] != 3:
            img = np.transpose(img, (1, 2, 0))

    if img.ndim == 2:
        img_channel = img
    elif img.ndim == 3:
        if img.shape[2] <= channel_index:
            return {"error": f"影像 channel 數量不足，無法取得索引 {channel_index}"}
        img_channel = img[:, :, channel_index]
    else:
        return {"error": "不支援的影像格式"}

    #--------------------------------------------------------------
    # 2. 擷取 ROI
    # --------------------------------------------------------------
    def extract_roi(image, kernel_size=128, visualize=True):
        H, W = image.shape
        non_zero_pixels = image[image > 0]
        if non_zero_pixels.size == 0:
            return None, None
        non_zero_ratio = non_zero_pixels.size / image.size
        non_zero_avg = np.mean(non_zero_pixels)
        threshold = non_zero_ratio * non_zero_avg * 0.5

        valid_blocks = []
        for y in range(0, H - kernel_size + 1, kernel_size):
            for x in range(0, W - kernel_size + 1, kernel_size):
                block = image[y:y + kernel_size, x:x + kernel_size]
                if block.mean() > threshold:
                    valid_blocks.append((x, y))
        if len(valid_blocks) == 0:
            return None, None
        valid_blocks = np.array(valid_blocks)

        y_vals = np.unique(valid_blocks[:, 1])
        edge_blocks = []
        left_max_x_center = None
        right_min_x_center = None

        for y in y_vals:
            row_blocks = valid_blocks[valid_blocks[:, 1] == y]
            sorted_row = row_blocks[np.argsort(row_blocks[:, 0])]
            left_blocks = sorted_row[:2]
            edge_blocks.extend([tuple(pt) for pt in left_blocks])
            if left_blocks.size > 0:
                left_max_x = np.max(left_blocks[0, :])
                left_max_x_center = left_max_x + kernel_size // 2
            if sorted_row.shape[0] > 2:
                right_blocks = sorted_row[-2:]
                edge_blocks.extend([tuple(pt) for pt in right_blocks])
                if right_blocks.size > 0:
                    right_min_x = np.min(right_blocks[:, 0])
                    right_min_x_center = right_min_x + kernel_size // 2

        y1 = int(H * 0.15)
        y2 = int(H * 0.85)

        if left_max_x_center is None or right_min_x_center is None:
            x1 = int(W * 0.10)
            x2 = int(W * 0.90)
            roi_image = image[y1:y2, x1:x2]
        else:
            roi_image = image[y1:y2, left_max_x_center:right_min_x_center]

        # 生成 ROI 擷取圖 (用於報告)
        if visualize:
            fig, ax = plt.subplots(1, 1, figsize=(15, 5))
            ax.imshow(image, cmap='gray')
            ax.set_title('Particle Distribution ROI Extraction')
            ax.axis('off')
            if left_max_x_center is not None and right_min_x_center is not None:
                for (x, y) in set(edge_blocks):
                    rect = plt.Rectangle((x, y), kernel_size, kernel_size,
                                         edgecolor='red', facecolor='none', linewidth=1)
                    ax.add_patch(rect)
                ax.axvline(left_max_x_center, linestyle='--', color='blue', linewidth=1)
                ax.axvline(right_min_x_center, linestyle='--', color='cyan', linewidth=1)
            else:
                x1 = int(W * 0.10)
                x2 = int(W * 0.90)
                ax.axvline(x1, linestyle='--', color='blue', linewidth=1)
                ax.axvline(x2, linestyle='--', color='cyan', linewidth=1)
            ax.axhline(y1, linestyle='--', color='green', linewidth=1)
            ax.axhline(y2, linestyle='--', color='green', linewidth=1)
            plt.tight_layout()

            # 保存視覺化結果
            roi_viz_path = os.path.join(output_dir, f"roi_visualization_{os.path.basename(image_path)}.png")
            plt.savefig(roi_viz_path)
            plt.close()

            return roi_image, roi_viz_path

        return roi_image, None

    #--------------------------------------------------------------
    # 3. 計算局部統計數據
    # --------------------------------------------------------------
    def local_stats(image, n_segments):
        H, W = image.shape
        step = int(W / n_segments)
        local_means = []
        local_stds = []
        for i in range(n_segments):
            start = i * step
            if start + H > W:
                break
            window = image[:, start:start + H]
            local_means.append(np.mean(window))
            local_stds.append(np.std(window))
        return np.array(local_means), np.array(local_stds)

    # 生成分布圖 (用於報告)
    def generate_distribution_plot(local_means, img_label, output_dir):
        plt.figure(figsize=(12, 6))
        plt.plot(local_means, marker='o', linestyle='-')
        plt.title(f'Particle Distribution Along Flow Channel - {img_label}')
        plt.xlabel('Flow Channel Segments')
        plt.ylabel('Local Mean Intensity')
        plt.grid(True, alpha=0.3)

        # 添加平均值水平線
        plt.axhline(y=np.mean(local_means), color='r', linestyle='--', alpha=0.7,
                   label=f'Average: {np.mean(local_means):.2f}')

        # 添加標準差範圍帶
        plt.fill_between(
            range(len(local_means)),
            np.mean(local_means) - np.std(local_means),
            np.mean(local_means) + np.std(local_means),
            color='r', alpha=0.1,
            label=f'Std Dev: {np.std(local_means):.2f}'
        )

        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"distribution_plot_{img_label}.png")
        plt.savefig(plot_path)
        plt.close()

        return plot_path

    roi_result = extract_roi(img_channel, visualize=True)
    if roi_result is None or roi_result[0] is None:
        return {"error": "ROI 擷取失敗"}

    roi, roi_viz_path = roi_result

    roi_norm = ((roi - roi.min()) / (roi.max() - roi.min()) * 255).astype(np.uint8)
    n_segments = 50
    local_means, local_stds = local_stats(roi_norm, n_segments)
    local_cvs = [s/m if m != 0 else 0 for m, s in zip(local_means, local_stds)]
    mean_of_means = np.mean(local_means)
    std_of_means = np.std(local_means)
    overall_cv = std_of_means / mean_of_means if mean_of_means != 0 else 0

    base_name = os.path.basename(image_path)
    img_label = base_name.split('_')[0].replace(" ", "")

    # 生成分布圖
    plot_path = generate_distribution_plot(local_means, img_label, output_dir)

    # 保存局部統計數據 CSV
    rows = []
    for idx, (lm, ls, cv) in enumerate(zip(local_means, local_stds, local_cvs)):
        rows.append({
            "Image": img_label,
            "Segment_Index": idx,
            "Local_Mean": lm,
            "Local_Std": ls,
            "Local_CV": cv
        })
    df_detail = pd.DataFrame(rows)
    csv_detail = os.path.join(output_dir, f"local_stats_{img_label}.csv")
    df_detail.to_csv(csv_detail, index=False)

    # 保存摘要統計數據 CSV
    summary = [{
        "Image": img_label,
        "Mean_of_Local_Mean": mean_of_means,
        "Std_of_Local_Mean": std_of_means,
        "CV_of_Local_Mean": overall_cv,
        "ROI_Visualization": roi_viz_path,
        "Distribution_Plot": plot_path
    }]
    df_summary = pd.DataFrame(summary)
    csv_summary = os.path.join(output_dir, f"summary_{img_label}.csv")
    df_summary.to_csv(csv_summary, index=False)

    # 返回分析結果字典
    result_dict = {
        "Image": img_label,
        "Local_Mean": float(mean_of_means),
        "Local_Std": float(std_of_means),
        "Local_CV": float(overall_cv),
        "Detail_CSV": csv_detail,
        "Summary_CSV": csv_summary,
        "ROI_Visualization": roi_viz_path,
        "Distribution_Plot": plot_path
    }
    return result_dict

# ---------------------------- Flask API 建構 ----------------------------
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "API is running"}), 200

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # 建立請求 ID 作為輸出目錄
        request_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(RESULTS_FOLDER, request_id)
        os.makedirs(output_dir, exist_ok=True)

        # 獲取通道參數
        channel = request.form.get('channel', 'red')

        # 處理上傳的圖像文件
        image_files = request.files.getlist('imageFiles')
        if not image_files:
            return jsonify({"error": "未提供圖像文件"}), 400

        # 處理每個圖像文件
        results = []
        chart_previews = []

        for image_file in image_files:
            if not image_file.filename:
                continue

            # 保存上傳的圖像
            filename = secure_filename(image_file.filename)
            image_path = os.path.join(output_dir, filename)
            image_file.save(image_path)

            # 執行分析
            result = analyze_partical_distribution(image_path, channel, output_dir)

            if "error" in result:
                results.append(result)
                continue

            results.append(result)

            # 為前端準備圖表預覽
            # ROI 視覺化圖
            with open(result["ROI_Visualization"], "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                chart_previews.append({
                    "image": f"data:image/png;base64,{img_data}",
                    "caption": f"樣本 {result['Image']} 的 ROI 擷取"
                })

            # 分布圖
            with open(result["Distribution_Plot"], "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                chart_previews.append({
                    "image": f"data:image/png;base64,{img_data}",
                    "caption": f"樣本 {result['Image']} 的粒子分布 (CV: {result['Local_CV']:.4f})"
                })

        # 生成匯總 CSV 文件，供 report_generator_api.py 使用
        summary_rows = []
        for r in results:
            if "error" in r:
                continue
            summary_rows.append({
                "Image": r["Image"],
                "Local_Mean": r["Local_Mean"],
                "Local_Std": r["Local_Std"],
                "Local_CV": r["Local_CV"],
                "ROI_Visualization": r["ROI_Visualization"],
                "Distribution_Plot": r["Distribution_Plot"]
            })

        if summary_rows:
            combined_summary_df = pd.DataFrame(summary_rows)
            combined_summary_path = os.path.join(output_dir, "combined_cv_results.csv")
            combined_summary_df.to_csv(combined_summary_path, index=False)
        else:
            combined_summary_path = None

        # 準備簡短的報告預覽文本
        report_preview = "粒子分布分析結果摘要:\n\n"
        for r in results:
            if "error" in r:
                report_preview += f"樣本 {r.get('Image', '未知')} 分析失敗: {r['error']}\n"
                continue
            report_preview += f"樣本 {r['Image']}:\n"
            report_preview += f"- 局部均值: {r['Local_Mean']:.4f}\n"
            report_preview += f"- 局部標準差: {r['Local_Std']:.4f}\n"
            report_preview += f"- 變異係數(CV): {r['Local_CV']:.4f}\n\n"

        # 構建響應
        response = {
            "message": "粒子分布分析完成",
            "results": results,
            "chartPreviews": chart_previews,
            "reportPreview": report_preview,
            "analysisId": request_id,
            "combinedCsvPath": f"{request_id}/combined_cv_results.csv" if summary_rows else None
        }

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    try:
        # 確保檔案路徑安全
        if '..' in filename or filename.startswith('/'):
            return jsonify({"error": "Invalid filename"}), 400

        # 構建完整的檔案路徑
        file_path = os.path.join(RESULTS_FOLDER, filename)
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404

        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))