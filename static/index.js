document.getElementById("table_multi_selection").style.display = "none";
document.getElementById("table_single_selection").style.display = "block";

function show_table_name_type() {
  let table_type = document.getElementById("table_type").value;
  if (table_type === "single_table") {
    document.getElementById("table_multi_selection").style.display = "none";
    document.getElementById("table_single_selection").style.display = "block";
  } else {
    document.getElementById("table_multi_selection").style.display = "block";
    document.getElementById("table_single_selection").style.display = "none";
  }
}

function metadata_toggle() {
  var ele = document.getElementById("metadata");
  if (ele.style.display === "block") {
    ele.style.display = "none";
  } else {
    ele.style.display = "block";
  }
}

function eval_log_toggle() {
  var ele = document.getElementById("logs");
  if (ele.style.display === "block") {
    ele.style.display = "none";
  } else {
    ele.style.display = "block";
  }
}

function gototop() {
  window.scrollTo({
    top: 0,
    behavior: "smooth",
  });
}
