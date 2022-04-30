import React, { useState, useEffect } from "react";
import { Grid, LinearProgress, Card, CardContent, Typography, Button, TextField } from '@mui/material';
import CheckCircleOutlineRoundedIcon from '@mui/icons-material/CheckCircleOutlineRounded';
import TuneRoundedIcon from '@mui/icons-material/TuneRounded';
import {useNavigate, Route} from "react-router-dom";


let progressN = 0
export default function Training() {
    let navigate = useNavigate();

    const [status, setStatus] = useState("Training");

    useEffect(() => {
        progressN += 1
        if (progressN===1){
        const res = fetch('/training'
        ).then(response => response.json()
        ).then(data => {setStatus(data.status); console.log(data.status)});
        }
    }, []);

    const onSubmit = async () => {
        navigate('/result', {replace:true});
    }
 
    return (
        <div className="App" style={{padding:"60px"}}>
            <Typography gutterBottom variant="h3" align="center" >
            Data Auto-Augmentation
            </Typography>
            <Card style={{ maxWidth: 900, padding: "10px 5px", margin: "0 auto" }}>
                <CardContent>
                    <Grid style={{padding:"30px"}}>
                    <Typography gutterBottom variant="h6" align="center" >
                        Our auto-augment learners are working hard to generate your data augmentation policy ...
                    </Typography>
                    </Grid>

                    {status==="Training" &&
                        <Grid style={{padding:"60px"}}>
                            <LinearProgress color="primary"/>
                            <LinearProgress color="primary" />
                            <LinearProgress color="primary" />
                            <LinearProgress color="primary" />
                        </Grid>
                    }

                    <Grid style={{padding:"50px"}}>
                    <Typography variant='h6'>
                        Current status: {status}
                    </Typography>
                    </Grid>
                    
                    {status==="Training is done!" &&
                        <Button
                                type="submit"
                                variant="contained"
                                color='primary'
                                size='large'
                                onClick={onSubmit}
                            >
                                Show Results
                        </Button>
                    }
                </CardContent>
            </Card>
                
        </div>
    )
}