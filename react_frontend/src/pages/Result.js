import React, { useState, useEffect } from "react";
import { Grid, Paper, Card, CardContent, Typography, Button, CardMedia } from '@mui/material';
import output from './output.png'
import {useNavigate, Route} from "react-router-dom";
import axios from 'axios'
import fileDownload from 'js-file-download'
import policy from './policy.txt'

function readTextFile(file)
{
    var rawFile = new XMLHttpRequest();
    rawFile.open("GET", file, false);
    rawFile.onreadystatechange = function ()
    {
        if(rawFile.readyState === 4)
        {
            if(rawFile.status === 200 || rawFile.status == 0)
            {
                var allText = rawFile.responseText;
                alert(allText);
            }
        }
    }
    rawFile.send(null);
}
const policy_txt = readTextFile("./policy.txt");
console.log('policy_txt',policy_txt)

export default function Result() {

    const handleClick = () => {
        axios.get('/result', {
            responseType: 'blob',
          })
        .then((res) => {
          fileDownload(res.data, 'policy.txt');
          console.log(res.data)
        })
      }

    const doc = <a href="https://autoaug.readthedocs.io/en/latest/howto/AutoAugment.html#how-to-use-a-autoaugment-object-to-apply-autoaugment-policies-to-datasets-objects" >documentation</a>;
    
    return (
        <div className="App" style={{padding:"60px"}}>
            <Typography gutterBottom variant="h3" align="center" >
            Data Auto-Augmentation
            </Typography>
            <Card style={{ maxWidth: 900, padding: "10px 5px", margin: "0 auto" }}>
                <CardContent>
                    <Typography gutterBottom variant="h5" align="left">
                        Here are the results from our auto-augment agent:
                    </Typography>
                    <Grid style={{padding:"30px"}} container spacing={4} alignItems="center">
                        <Grid xs={7} item> 
                            <CardMedia
                            style={{width: "auto", maxHeight: "300px"}}
                            component="img"
                            image={output}
                            />
                        </Grid>
                        <Grid xs={5} item> 
                            <Paper elevation={3}>
                                {policy}
                                {/* <Typography>
                                write something here to explain the meaning of the graph to the user
                                </Typography>    */}
                            </Paper>
                            {/* <Typography>
                                write something here to explain the meaning of the graph to the user
                            </Typography> */}
                        </Grid>
                    </Grid>

                    <Typography  variant='subtitle1' align='center'>
                        You can download the augentation policy here
                    </Typography>

                    <Grid style={{padding:'5px'}}>
                        <Typography gutterBottom>
                            Please follow our {doc} to apply this policy to your dataset.
                        </Typography>
                    </Grid>

                    <Button
                            type="submit"
                            variant="contained"
                            color='primary'
                            size='large'
                            onClick={() => handleClick()}
                        >
                            Download
                    </Button>

                    
                </CardContent>
            </Card>

        </div>
    )
}