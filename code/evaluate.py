import numpy as np


def test_perf(prediction, gt_test, mask):
    case_count = prediction.shape[1] * prediction.shape[0]
    tes_mse = np.sum((prediction - gt_test) ** 2) / case_count
    # fff = np.sum((prediction - gt_test) ** 2)

    text_case_count = np.sum(mask)
    if text_case_count == 0:
        tes_mse_text = 0
    else:
        # aaa = (prediction - gt_test)
        # bbb = aaa * mask
        # ccc = bbb ** 2
        # ddd = ((prediction - gt_test) * mask) ** 2
        # eee = np.sum(ccc)
        tes_mse_text = np.sum(((prediction - gt_test) * mask) ** 2) / text_case_count

    price_case_count = case_count - text_case_count
    price_mask = np.abs(1 - mask)
    if price_case_count == 0:
        tes_mse_price = 0
    else:
        # aaa1 = (prediction - gt_test)
        # bbb1 = aaa1 * price_mask
        # ccc1 = bbb1 ** 2
        # ddd1 = ((prediction - gt_test) * mask) ** 2
        # eee1 = np.sum(ccc1)
        tes_mse_price = np.sum(((prediction - gt_test) * price_mask) ** 2) / price_case_count
    return tes_mse, tes_mse_text, tes_mse_price