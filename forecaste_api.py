from flask import Flask, request, jsonify
from patchmixer import Model
import torch


app = Flask(__name__)
@app.route("/configure", methods=["GET"])
def configure():
    try:
        # get data from the request
        data = request.get_json()
        enc_in = data.get("enc_in")
        seq_len = data.get("seq_len")
        pred_len = data.get("pred_len")
        patch_len = data.get("patch_len")
        stride = data.get("stride")
        padding_patch = data.get("padding_patch")

        return {
            "enc_in": enc_in,
            "seq_len": seq_len,
            "pred_len": pred_len,
            "patch_len": patch_len,
            "stride": stride,
            "padding_patch": padding_patch,
        }

    except Exception as e:
        return jsonify({"error:": str(e)}), 500


@app.route("/forward", methods=["POST"])
def forward():
    try:
        configs = configure()

        model = Model(configs)
        x, batch_x_mark, dec_inp, batch_y_mark = (
            torch.rand(),
            None,
            None,
            None,
        )  # todo: get accurate values
        model.forward(x, batch_x_mark, dec_inp, batch_y_mark)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
