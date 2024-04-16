from flask import Flask, render_template, redirect, request
from table_maker import (
    get_demo_table,
    get_synthetic_table,
    get_eval_reports,
    get_diag_reports,
)
import os

# __name__ == '__main__'
app = Flask(__name__)


@app.route("/")
def Home():
    return render_template("index.html")


@app.route("/home")
def Home_other():
    return redirect("/")


@app.route("/generate_synthetic_table", methods=["POST"])
def submit_data():
    data, synthetic_table, metadata, eval_reports, diag_reports = (
        {},
        None,
        None,
        None,
        None,
    )

    model_dict = {
        "clf1": "GaussianCopulaSynthesizer",
        "clf2": "CopulaGanSynthesizer",
        "clf3": "CTGanSynthesizer",
        "clf4": "TVAESynthesizer",
    }
    if request.method == "POST":
        form_data = request.form
        table_type = form_data.get("table_type")
        num_rows = int(form_data.get("select_num_rows"))
        data["table_type"] = table_type
        data["num_rows"] = num_rows
        table_name = ""
        if table_type == "single_table":
            table_name = form_data.get("table_single_selection")
            data["table_name"] = table_name
            # show demo table
            demo_table = get_demo_table(table_type, table_name)

            # show synthetic generated table
            synthetic_table, metadata, selected_model, eval_logs, eval_reports = (
                get_synthetic_table(
                    table_type=table_type, table_name=table_name, num_rows=num_rows
                )
            )

            diag_reports = get_diag_reports(table_type, table_name)

            data["model"] = model_dict[selected_model]

            files = os.listdir("static")
            for f in files:
                if f.endswith(".png"):
                    os.remove(os.path.join("static", f))
            metadata.visualize(
                show_table_details="summarized",
                output_filepath="./static/my_metadata_summarized.png",
            )
            metadata.visualize(
                show_table_details="full",
                output_filepath="./static/my_metadata_full.png",
            )

            return render_template(
                "index.html",
                data=data,
                demo_table=demo_table,
                synthetic_table=synthetic_table,
                metadata=metadata,
                diag_reports=diag_reports,
                eval_reports=eval_reports,
                eval_logs=eval_logs,
            )

        else:
            table_name = form_data.get("table_multi_selection")
            data["table_name"] = table_name
            demo_table = get_demo_table(table_type, table_name)

        # show reports

    return render_template(
        "index.html",
        type_table_name=data,
        demo_table=demo_table,
        synthetic_table=synthetic_table,
        metadata=metadata,
        diag_reports=diag_reports,
        eval_reports=eval_reports,
    )


if __name__ == "__main__":
    # app.debug=True
    app.run(debug=True)
