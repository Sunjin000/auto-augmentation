import React, { useState, useEffect } from "react";
import { Grid, Paper, Card, CardContent, Typography, Button, CardMedia, Box } from '@mui/material';
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
                    <Box 
                                    display="flex" 
                                    justifyContent="center"
                                    style={{padding:"30px"}}
                                >
                    <Grid style={{width:"400px"}}>
                    <Paper elevation={3}>
                    <CardMedia
                            style={{width: "auto", maxHeight: "300px", }}
                            component="img"
                            image={output}
                            />
                    </Paper>
                    </Grid>
                    </Box>

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