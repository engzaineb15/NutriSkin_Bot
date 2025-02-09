function analyzeImage() {
    let fileInput = document.getElementById("imageInput");
    let file = fileInput.files[0];
    
    if (!file) {
        alert("يرجى اختيار صورة!");
        return;
    }

    let formData = new FormData();
    formData.append("image", file);

    fetch("/analyze", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText = data.diagnosis;
    })
    .catch(error => {
        console.error("Error:", error);
        document.getElementById("result").innerText = "خطأ في التحليل!";
    });
}
