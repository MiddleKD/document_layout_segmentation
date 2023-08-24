import numpy as np
from scipy.ndimage.measurements import label

def get_bboxes_from_segmentation(segmentation_map):
    unique_labels = np.unique(segmentation_map)
    bboxes = []
    
    for u_label in unique_labels:
        if u_label == 0:  # 배경은 무시
            continue
            
        # 특정 레이블에 대한 이진 마스크 생성
        binary_mask = (segmentation_map == u_label).astype(np.int64)
        
        # 연결된 구성요소 라벨링 수행
        labeled, num_features = label(binary_mask)

        for i in range(1, num_features + 1):
            rows, cols = np.where(labeled == i)
            
            if len(rows) == 0: continue
            
            min_row, max_row = min(rows), max(rows)
            min_col, max_col = min(cols), max(cols)
            
            bboxes.append((min_col, min_row, max_col, max_row))
        
    return bboxes


if __name__ == "__main__":
    # 예제
    segmentation_map = np.array([
        [1, 1, 0, 0, 2, 2],
        [1, 1, 0, 0, 2, 2],
        [0, 0, 0, 0, 0, 0],
        [3, 3, 0, 4, 1, 4],
        [3, 3, 0, 4, 1, 4],
    ])

    bboxes = get_bboxes_from_segmentation(segmentation_map)
    print(bboxes)