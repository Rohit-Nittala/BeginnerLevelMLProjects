/*const BACKGROUND_COLOUR = '#000000';*/
/*LINE_COLOR = '#FFFFFF';
LINE_WIDTH = 15;*/

var currentX = 0;
var currentY = 0;
var previousX = 0;
var previousY = 0;

var canvas;
var context;

function prepareCanvas() {
    console.log('Preparing Canvas');
    canvas = document.getElementById('my-canvas');
    context = canvas.getContext('2d');
    context.fillStyle = "#000000";
    context.fillRect(0,0, canvas.clientWidth, canvas.clientHeight);

    context.strokeStyle = "#BCFF00";
    context.lineWidth  = 15;
    context.lineJoin = 'round';
    //console.log('Filled style.');

    isPainting = false;

    document.addEventListener('mousedown', function(event){
        //console.log('Mouse Pressed');
        isPainting = true;
        currentX = event.clientX - canvas.offsetLeft;
        currentY = event.clientY - canvas.offsetTop;


    });

    document.addEventListener('mousemove', function(event){
        /*console.log('Moved!');*/
        if (isPainting){
            previousX = currentX;
            currentX = event.clientX - canvas.offsetLeft;

            previousY = currentY;
            currentY = event.clientY - canvas.offsetTop;

            draw();
        }
        
    });
    
    document.addEventListener('mouseup', function(event){
        //console.log('Mouse Released!');
        isPainting = false;
    });

    canvas.addEventListener('mouseleave', function(event){
        isPainting = false;
    });

    //Touch Events
    canvas.addEventListener('touchstart', function(event){
        //console.log('TouchDown!');
        isPainting = true;
        currentX = event.touches[0].clientX - canvas.offsetLeft;
        currentY = event.touches[0].clientY - canvas.offsetTop;
    });
    canvas.addEventListener('touchend', function(event){
        isPainting = false;
    });
    canvas.addEventListener('touchcancel', function(event){
        isPainting = false;
    });
    canvas.addEventListener('touchmove', function(event){
        /*console.log('Moved!');*/
        if (isPainting){
            previousX = currentX;
            currentX = event.touches[0].clientX - canvas.offsetLeft;

            previousY = currentY;
            currentY = event.touches[0].clientY - canvas.offsetTop;

            draw();
        }
        
    });
}

function draw() {
    context.beginPath();
    context.moveTo(previousX, previousY);
    context.lineTo(currentX, currentY);
    context.closePath();
    context.stroke();
}

function clearCanvas(){
    var currentX = 0;
    var currentY = 0;
    var previousX = 0;
    var previousY = 0;
    context.fillRect(0,0, canvas.clientWidth, canvas.clientHeight);


}