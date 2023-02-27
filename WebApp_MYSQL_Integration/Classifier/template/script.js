const loadFile = function (event) {
  console.log("Hello world");
  var image = document.getElementById("output");
  var condition = document.getElementById("div_to_remove");
  console.log(condition);
  condition.innerHTML = "";
  image.src = URL.createObjectURL(event.target.files[0]);
};
