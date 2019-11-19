var unlock = ()=>{
	var e = document.getElementsByClassName("lock")[0];
	e.classList.toggle("unlocked");
	var p = document.getElementById("lock-text");
	p.innerHTML = "UNLOCK";
	setTimeout(fadeOutEffect, 1000, "lock-container", 200);
};

var wrong = (...args)=>{
	var warning = document.createElement("div");
	warning.classList.add("blink");
	args.forEach((t)=>{
		var text = document.createElement("p");
		text.innerHTML = t;
		warning.appendChild(text);
	});
	
	document.body.appendChild(warning);
	return warning;
};

function fadeOutEffect(id, i) {
    var fadeTarget = document.getElementById(id);
    var fadeEffect = setInterval(function () {
        if (!fadeTarget.style.opacity) {
            fadeTarget.style.opacity = 1;
        }
        if (fadeTarget.style.opacity > 0) {
            fadeTarget.style.opacity -= 0.1;
        } else {
        	fadeTarget.remove();
            clearInterval(fadeEffect);

        }
    }, i);
}