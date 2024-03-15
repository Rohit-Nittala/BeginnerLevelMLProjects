var model;

async function loadModel() {
    model = await tf.loadGraphModel('TFJS/model.json')
    /*console.log(model);*/
    //const result = model.predict(myTensor);
    //result.print();
  }

function predictImage() {
    //console.log("Processing...");
    
    //Loading our Image from the canvas and coinverting it to black and white
    let image = cv.imread(canvas);
    cv.cvtColor(image, image, cv.COLOR_RGBA2GRAY, 0);
    cv.threshold(image, image, 175, 255, cv.THRESH_BINARY);

    //Finding the exact boundry.
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    cv.findContours(image, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);

    //Drawing bounding Rectangle.
    let cnt = contours.get(0);
    let rect = cv.boundingRect(cnt);
    image = image.roi(rect);


    //Resizing.
    var height = image.rows;
    var width = image.cols;
    //console.log('Beofre resizing', width, height);
    
    if (height > width){
        height = 20;
        const scaleFactor = image.rows / height;
        width = Math.round(image.cols / scaleFactor);
    } else {
        width = 20;
        const scaleFactor = image.cols / width;
        height = Math.round(image.rows / scaleFactor);
    }

    //console.log('After resizing', width, height);

    let newsize = new cv.Size(width, height);
    cv.resize(image, image, newsize, 0, 0, cv.INTER_AREA)

    // Adding Padding.
    const LEFT = Math.ceil(4 + (20 - width) / 2);
    const RIGHT = Math.floor(4 + (20 - width) /2);
    const TOP = Math.ceil(4 + (20 - height) / 2);
    const BOTTOM = Math.floor(4 + (20 - height) / 2);

    const BLACK = new cv.Scalar(0,0,0,0);
    cv.copyMakeBorder(image, image, TOP, BOTTOM, LEFT, RIGHT, cv.BORDER_CONSTANT, BLACK);
    /*console.log('After padding', LEFT, RIGHT, TOP, BOTTOM);
    console.log('After all effects:', image.rows, image.cols);*/

    // Center of Mass
    cv.findContours(image, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
    cnt = contours.get(0);
    const Moments = cv.moments(cnt, false);
    const cx = Moments.m10 / Moments.m00;
    const cy = Moments.m01 / Moments.m00;

    //console.log(Moments.m00, cx, cy);

    // Shifting the image
    const X_SHIFT = Math.round(image.cols / 2.0 - cx);
    const Y_SHIFT = Math.round(image.rows / 2.0 - cy);

    newSize = new cv.Size(image.cols, image.rows);
    const M = new cv.matFromArray(2, 3, cv.CV_64FC1, [1, 0, X_SHIFT, 0, 1, Y_SHIFT]);
    cv.warpAffine(image, image, M, newSize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, BLACK);


    let pixelValues = image.data; 
    
    pixelValues = Float32Array.from(pixelValues);

    pixelValues = pixelValues.map(function(item) {
        return item/255.0;
    });

    // Testing 
    /*const outputCanvas = document.createElement('CANVAS');
    cv.imshow(outputCanvas, image);
    document.body.appendChild(outputCanvas);*/

    const X = tf.tensor([pixelValues]);

    const result = model.predict(X);
    result.print();
    const output = result.dataSync()[0];


    //Cleanup
    image.delete();
    contours.delete();
    cnt.delete();
    hierarchy.delete();
    M.delete();
    X.dispose();
    result.dispose();

    return output;
}