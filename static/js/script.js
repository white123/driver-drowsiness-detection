var socket = io();
var wrong_driver = null;
var danger = null;
var lock = true;
var name = "";


socket.on('connect', function() {
  console.log("Connected")
});
socket.on('server_response', function(res) {
  var ret = JSON.parse(res);
  if(ret.driving_status == "Driving Normally"){
  	if(lock){
	  	unlock();
      name = ret.name;
	  	lock = false;
  	}else if(wrong_driver){
  		wrong_driver.remove();
  		wrong_driver = null;
      limit = false;
  	}
  }else if(ret.driving_status == "Wrong Driver"){
  	if(!wrong_driver && !danger){
      limit = true;
  		wrong_driver = wrong(ret.driving_status, "Speed limited to 20km/h");
    }
  }
  console.log(ret.safety_status);
  if(ret.safety_status){
    var e = document.getElementById("safe-status");
    e.innerHTML = ret.safety_status;
    e.style.color = {"Danger":"red", "Warning":"yellow", "Safe":"green"}[ret.safety_status];
  }
  if(ret.safety_status == "Danger"){
    if(!wrong_driver && !danger)
      danger = wrong("Hey, "+name, "Please be concentrate!")
  }else if(danger){
      danger.remove();
      danger = null;
  }
  if(ret.theta)
  	addHead(ret.theta);
});