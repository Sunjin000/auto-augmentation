import React, { useState, useEffect } from "react";
import { Grid, List, ListItem, Avatar, ListItemAvatar, ListItemText, Card, CardContent, Typography, Button, CardMedia } from '@mui/material';
import output from './output.png'
import {useNavigate, Route} from "react-router-dom";
import axios from 'axios'
import fileDownload from 'js-file-download'

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
                        <Grid xs={6} item> 
                            <img src={output} alt='output' />
                        </Grid>
                        <Grid xs={6} item> 
                            <Typography>
                                write something here to explain the meaning of the graph to the user
                            </Typography>
                        </Grid>
                    </Grid>

                    <Typography gutterBottom variant='h6' align='center'>
                        You can download the augentation policy here
                    </Typography>

                    <Button
                            type="submit"
                            variant="contained"
                            color='primary'
                            size='large'
                            onClick={() => handleClick('https://avatars.githubusercontent.com/u/9919?s=280&v=4', 'sample')}
                        >
                            Download
                    </Button>

                    <Grid style={{padding:'10px'}}>
                        <Typography>
                            Please follow our documentation to apply this policy to your dataset.
                        </Typography>
                    </Grid>
                    
                </CardContent>
            </Card>

        </div>
    )
}