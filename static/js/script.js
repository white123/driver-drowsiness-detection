var socket = io();
var warning = null;
var warning_type = null;
var lock = true;
var name = "";


socket.on('connect', function() {
  console.log("Connected")
});
socket.on('server_response', function(res) {
  var ret = JSON.parse(res);
  console.log(ret.headpose_status, typeof(ret.headpose_status), warning);
  if(lock){
    if(ret.driving_status == "Driving Normally"){
    	unlock();
      name = ret.name;
    	lock = false;
    }else{
      return;
    }
  }

  //warning system
  limit = false;
  if(ret.driving_status == "Invalid Driver"){
    limit = true;
    if(warning_type != "driving" && warning){
      warning.remove();
      warning = null;
    }
  	if(!warning){
  		warning = wrong(ret.driving_status, "Speed limited to 20km/h");
      warning_type = "driving";
    }
  }else if(ret.safety_status == "Danger"){
    if(warning_type != "safety" && warning){
      warning.remove();
      warning = null;
    }
    if(!warning){
      warning = wrong("Hey, "+name, "Please be concentrate!");
      warning_type = "safety";
    }
  }else if(ret.headpose_status != undefined){
    switch(ret.headpose_status){
      case 0:
        if(warning_type != "down" && warning){
          warning.remove();
          warning = null;
        }
        if(!warning){
          warning = wrong("Hey, "+name, "Please be concentrate!");
          warning_type = "down";
        }
        break;
      case 1:
      case 2:
        if(warning_type != "forward" && warning){
          warning.remove();
          warning = null;
        }
        if(!warning){
          warning = wrong("Hey, "+name, "Please look forward!");
          warning_type = "forward";
        }
        break;
      case 3:
        if(warning_type != "left-light" && warning){
          warning.remove();
          warning = null;
        }
        if(!warning){
          warning = light("left");
          warning_type = "left-light";
        }
        break;
      case 4:
        if(warning_type != "right-light" && warning){
          warning.remove();
          warning = null;
        }
        if(!warning){
          warning = light("right");
          warning_type = "right-light";
        }
        break;
      case 5:
      default:
        if(warning){
          warning.remove();
          warning = null;
          warning_type = null;
        }
        break;
    }
  }else if(warning){
      warning.remove();
      warning = null;
      warning_type = null;
  }

  //update status text
  if(ret.safety_status){
    var e = document.getElementById("safe-status");
    e.innerHTML = ret.safety_status;
    e.style.color = {"Danger":"red", "Warning":"yellow", "Safe":"green"}[ret.safety_status];
  }
  
  //update line chart
  if(ret.theta)
  	addHead(ret.theta);
});