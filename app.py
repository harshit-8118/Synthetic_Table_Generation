from flask import Flask, render_template, redirect, request
from table_maker import (
    get_demo_table,
    get_synthetic_table,
    # get_eval_reports,
    get_diag_reports,
    get_evaluation_graphs,
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
    data, synthetic_table, metadata, eval_reports, diag_reports, eval_logs = (
        {},
        None,
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
        table_type, num_rows = form_data.get("table_type"), int(
            form_data.get("select_num_rows")
        )
        data["table_type"] = table_type
        data["num_rows"] = num_rows
        table_name = ""
        if table_type == "single_table":
            table_name = form_data.get("table_single_selection")
            data["table_name"] = table_name

            # get demo table
            demo_table = get_demo_table(table_type, table_name)

            # get synthetic generated table, model, evaluation logs, evaluation reports and metadata
            synthetic_table, metadata, selected_model, eval_logs, eval_reports = (
                get_synthetic_table(
                    table_type=table_type, table_name=table_name, num_rows=num_rows
                )
            )

            # get diagnostic reports
            diag_reports = get_diag_reports(table_type, table_name)

            data["model"] = model_dict[selected_model]

            # clear cached visualization images
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

            # get plotly_graph for single table
            graph_path = get_evaluation_graphs(table_type, table_name)
            if graph_path:
                graph_path = graph_path[:5]

            return render_template(
                "index.html",
                data=data,
                demo_table=demo_table,
                synthetic_table=synthetic_table,
                metadata=metadata,
                diag_reports=diag_reports,
                eval_reports=eval_reports,
                eval_logs=eval_logs,
                graph_path=graph_path,
            )

        else:
            table_name = form_data.get("table_multi_selection")
            data["table_name"] = table_name

            # show demo table
            demo_table = get_demo_table(table_type, table_name)

            # show synthetic generated table
            synthetic_table, metadata = get_synthetic_table(
                table_type=table_type, table_name=table_name, num_rows=num_rows
            )

            # get_eval_reports(table_type=table_type, table_name=table_name)
            # diag_reports = get_diag_reports(table_type, table_name)

            # clear cached visualization images
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


if __name__ == "__main__":
    app.run(debug=True)
