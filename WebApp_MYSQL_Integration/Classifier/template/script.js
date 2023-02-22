const loadFile = function (event) {
  // console.log("Hello world");
  var image = document.getElementById("output");
  image.src = URL.createObjectURL(event.target.files[0]);
};
