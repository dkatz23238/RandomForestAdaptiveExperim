var connect = require('connect');
var fs = require('fs');
var JSONDATA = JSON.parse(fs.readFileSync('/home/david/Desktop/ax-container/app/experiment-results/2019-05-09T13:31:55.488260.json', 'utf8'));
var cons = require('consolidate');

// view engine setup
var express = require('express');
var app = express();
var path = require('path');
app.engine('html', cons.swig)
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'html');

// viewed at http://localhost:8080
app.get('/', function(req, res) {
    let path = __dirname + "/index.html";
    res.sendfile(path)

});

app.listen(8080);


