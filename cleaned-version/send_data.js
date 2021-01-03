const spawn = require('child_process').spawn;

const scriptExecution = spawn("python", ["pipe.py"]);

var i = 0;


var data = " hema "+i

// Handle normal output
scriptExecution.stdout.on('data', (data) => {
console.log(String.fromCharCode.apply(null, data));
});

// Write data (remember to send only strings or numbers, otherwhise python wont understand)

scriptExecution.stdin.write(data);
// End data write

i++;
scriptExecution.stdin.end();



