//Global variable
var base64data
//Take permission to record audio and then send stream
navigator.mediaDevices.getUserMedia({audio:true,video: true})
      .then(stream => {handlerFunction(stream)
                       });

//Function to handle the input stream
          function handlerFunction(stream) {
            //Stores the stream
            window.rec = new MediaRecorder(stream);
            // Task when recording starts
            var gumVideo = document.querySelector('video#gum');
            gumVideo.srcObject = stream;
            gumVideo.style.display = "none";
            rec.onstart = e =>{

            totalaudio = [];

            }

            //Task when audio is inputted into the stream
            rec.ondataavailable = e => {

              //Add data to the audio buffer
              totalaudio.push(e.data);

            }

            //Task when audio is stopped
            rec.onstop = e =>{
              //Print length of buffer
              console.log(totalaudio.length)
              //Create blob of the new audio
              let blob = new Blob(totalaudio);
              //Send blob to the backend
              sendData(blob)
              //recordedAudio.src = URL.createObjectURL(blob);
              //recordedAudio.controls=true;
              //recordedAudio.autoplay=false;

            }
            //Interval function to interrupt the recording every 5 seconds.
            setInterval(function(){
              //If the recording has started, interrupt.
              if(rec.state == "recording"){

                rec.stop()
                totalaudio = []
                rec.start()

              }
              //6000 as it is always a second less = 5000ms
            }, 6000);

           }



          //POST data out
          async function sendData(data) {

             console.log(data)
             //Upload the blob to a file reader
             var reader = new FileReader();
             reader.readAsDataURL(data);
             //Converting it to a base64 string.
             reader.onload = function() {
               // console.log("reader onload ");
                  base64data = reader.result.split(',')[1];
                  //console.log(base64data);
                  socket.emit("blob event",base64data)

             }



          /*    var result = await response.json();
              console.log(result);
              console.log(result.videodata)
              //console.log(typeof(result.probability));
              //console.log(typeof(parseFloat(result.probability)));
              var pred = document.getElementById("prediction");
              var element = document.getElementById("container1");
              var vd = document.getElementById('outvdg')
              //im.src = url_for('static',{"filename":result.vdagedata});
              if (stopRecord.disabled == false){
              vd.src = result.videodata;
              vd.style.display = "block";
            }
              // Add an event listener
              pred.addEventListener("Trigger", function(e) {
                console.log(e.detail.message); // Prints "Example of an event"
                pred.innerHTML =" ";
                element.style.display ="block";

              });

              // Create the event
              var event = new CustomEvent("Trigger", { "detail": {
                message : "You Coughed!",
                } });

              if ((result.label === "coughing")&&(parseFloat(result.probability) >= 0.90)){
                pred.dispatchEvent(event);
              }
              else{
                if (stopRecord.disabled == true){
                  pred.innerHTML = ""
                }
                else
                {
                pred.innerHTML = "Detecting...";
                }
                element.style.display = "none";
              }*/
            }

         // When start button is clicked.
      window.onload = function(){
        record.onclick = e => {
          let pred = document.getElementById("prediction");
          let st = document.getElementById("start");
          let sp = document.getElementById("stop");
          let gumVideo = document.querySelector('video#gum');
          st.innerHTML = "Started recording"
          st.style.backgroundColor = "white"
          sp.innerHTML = ""
          pred.innerHTML = "Detecting..."
          gumVideo.style.display = "block"

          console.log('Start:I was clicked')
          record.disabled = true;
          record.style.backgroundColor = "grey"
          stopRecord.disabled=false;

          rec.start();

        }

        //When stop button is clicked.
        stopRecord.onclick = e => {
          let pred = document.getElementById("prediction");
          let st = document.getElementById("start");
          let sp = document.getElementById("stop");
          let gumVideo = document.querySelector('video#gum');
          st.innerHTML = ""
          sp.innerHTML = "Stopped recording"
          sp.style.backgroundColor = "#f4623a"
          pred.innerHTML = ""
          gumVideo.style.display = "none"

          console.log("Stop:I was clicked")
          record.disabled = false;
          stopRecord.disabled=true;
          record.style.backgroundColor = "white"
          rec.stop();

        }
      }
