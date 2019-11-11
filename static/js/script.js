var socket = io();
socket.on('connect', function() {
  console.log("Connected")
});
socket.on('server_response', function(res) {
  var ret = JSON.parse(res);
  if(ret.theta)
  	addHead(ret.theta);
});