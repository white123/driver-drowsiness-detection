var addHead = ([h, v])=>{
	if(window.headpose.data.datasets[0].data.length >= 30)
		window.headpose.data.datasets[0].data.shift();
	if(window.headpose.data.datasets[1].data.length >= 30)
		window.headpose.data.datasets[1].data.shift();
	window.headpose.data.datasets[0].data.push(Math.round(h));
	window.headpose.data.datasets[1].data.push(Math.round(v));
	window.headpose.update();
}

var config = {
    type: 'line',
    data: {
        labels: Array(30).fill(0),
        datasets: [{
            label: 'Horizontal',
            backgroundColor: window.chartColors.red,
            borderColor: window.chartColors.red,
            fill: false,
            data: [],
        }, {
            label: 'Verticle',
            backgroundColor: window.chartColors.blue,
            borderColor: window.chartColors.blue,
            fill: false,
            data: [],
        }]
    },
    options: {
        responsive: true,
        title: {
            display: true,
            text: 'Headpose',
            fontColor: "white"
        },
        legend: {
            labels: {
                fontColor: "white",
            }
        },
        scales: {
            xAxes: [{
                display: false,
            }],
            yAxes: [{
                display: false,
	            ticks: {
			        beginAtZero:true,
			        mirror:false,
			        suggestedMin: -30,
			        suggestedMax: 30,
			    }
			}]
        }
    }
};

(function() {
    var ctx = document.getElementById('headpose').getContext('2d');
    window.headpose = new Chart(ctx, config);
})();