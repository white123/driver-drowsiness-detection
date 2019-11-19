var speed = 0;
function update(){
    speed -= 0.25;
    if(limit && speed > 20) speed -= 1;
    if (speed < 0) speed = 0;
	var e = document.getElementById("speed");
	e.innerHTML = Math.round(speed).toString().padStart(2, '0');
	setTimeout(update, 100);
}

update();

document.addEventListener('keydown', function(event) {
    if(event.key == "ArrowDown" || event.key == 's') {
        speed -= 2;
        if (speed < 0) speed = 0;
    }
    else if(event.key == "ArrowUp" || event.key == 'w') {
        if(!(limit && speed > 20))
            speed += 2;
        if (speed > 150) speed = 150;
    }
});