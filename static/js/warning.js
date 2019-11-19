var warn = (t)=>{
	var warning = document.createElement("div");
	warning.id = "notification";
	warning.classList.add("warning");
	warning.innerHTML=t;
	document.body.appendChild(warning);
	setTimeout(fadeOutEffect, 1000, "notification", 50);
};

var accept = (t)=>{
	var warning = document.createElement("div");
	warning.id = "notification";
	warning.classList.add("accept");
	warning.innerHTML=t;
	document.body.appendChild(warning);
	setTimeout(fadeOutEffect, 1000, "notification", 50);
};