var light = (direc) =>{
	li = null;
	if(direc == "left"){
		li = document.createElement("div");
		li.id = "left-light";
		document.body.appendChild(li);
	}else if(direc == "right"){
		li = document.createElement("div");
		li.id = "right-light";
		document.body.appendChild(li);
	}
	return li;
}