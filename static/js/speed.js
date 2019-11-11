var speed = 0;
function update(){
	var e = document.getElementById("speed");
	e.innerHTML = speed.toString().padStart(2, '0');
	//setTimeout(update, 100);
}

update();

document.addEventListener('keydown', function(event) {
	
    if(event.key == "ArrowDown") {
        speed -= 1;
        if (speed < 0) speed = 0;
        update();
    }
    else if(event.key == "ArrowUp") {
        speed += 1;
        update();
    }
});