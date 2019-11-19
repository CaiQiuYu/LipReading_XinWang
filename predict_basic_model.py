import torch
import pickle
from model.BasicModel import LipModel
from utils import get_train_data


print("======> Load Model")
model_path = 'weights/basic/bagging_no_att_1.pt'
device = 'cuda:0'
model = LipModel()

with open(model_path, 'rb') as f:
    model.load_state_dict(torch.load(f, map_location=device))
model.eval()
model.to(device)


index2label = []
with open('dictionary/dictionary.txt', 'r', encoding='utf-8') as f:
    for word in f:
        index2label.append(word.split(',')[0])

with open('data_pkl/data_test.pkl', 'rb') as f:
    test_data = pickle.load(f)
    test_ids = pickle.load(f)
test_data, test_ids = get_train_data(test_data, test_ids, 16, test_data=True)

print('=======>> Predict')
predict_result = []
with torch.no_grad():
    for idx in range(len(test_data)):
        inputs = test_data[idx].to(device)
        possibility = model(inputs)[0]

        pred = torch.argmax(possibility, dim=-1).tolist()
        assert len(pred) == len(test_ids[idx])
        for i, ids in enumerate(test_ids[idx]):
            predict_result.append(ids + ',' + index2label[pred[i]])

with open('results/basic/basic_submit_1.csv', 'w', encoding='utf-8') as f:
    for line in predict_result:
        f.write(line + '\n')
