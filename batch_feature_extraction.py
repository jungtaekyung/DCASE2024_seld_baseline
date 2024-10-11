# Extracts the features, labels, and normalizes the development and evaluation split features.
# features, labels를  추출하고. 개발 및 평가 분할 기능을 정규화합니다.                           
import cls_feature_class
import parameters
import sys


def main(argv):  # parameter.py 파일에 제공된 구성에 해당하는 하나의 입력(task-id)이 필요합니다.                            
    # Expects one input - task-id - corresponding to the configuration given in the parameter.py file. 
    # Extracts features and labels relevant for the task-id  task-id 와 관련된 기능 및 레이블을 추출합니다.                 
    # It is enough to compute the feature and labels once. feature와 label을 한 번만 계산해도 충분합니다.                            

    # use parameter set defined by user   , one-line if 문
    task_id = '1' if len(argv) < 2 else argv[1]   # len(argv) < 2 == True , task_id='1' , len(argv) < 2 == False , task_id = argv[1]
    params = parameters.get_params(task_id)

    # -------------- Extract features and labels for development set -----------------------------
    if params['mode'] == 'dev':   # 'dev' - development or 'eval' - evaluation dataset
        dev_feat_cls = cls_feature_class.FeatureClass(params, is_eval=False)

        # # Extract features and normalize them
        dev_feat_cls.extract_all_feature()
        dev_feat_cls.preprocess_features()

        # # Extract labels
        dev_feat_cls.extract_all_labels()

        # # Extract visual features
        if params['modality'] == 'audio_visual':
            dev_feat_cls.extract_visual_features()

    else:
        dev_feat_cls = cls_feature_class.FeatureClass(params, is_eval=True)

        # # Extract features and normalize them
        dev_feat_cls.extract_all_feature()
        dev_feat_cls.preprocess_features()

        # # Extract visual features
        if params['modality'] == 'audio_visual':
            dev_feat_cls.extract_visual_features()


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)

