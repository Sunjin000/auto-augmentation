import React, { useState, useEffect } from "react";
import { Grid, CardMedia, Card, CardContent, Typography, Button, TextField } from '@mui/material';
import {Link, useNavigate, Route} from "react-router-dom";
import datafolder from "./dataset_upload.png"


const home_button = (
    <Button variant="contained"
        size='large'
        as={Link}
        to="/"
        >
            Home
    </Button>
);

export default function Home() {
    const [errorMsg, setErrorMsg] = useState();
    const [errorType, setErrorType] = useState();
    useEffect(() => {
        const res = fetch('/home').then(
          response => response.json()
          ).then(data => {setErrorMsg(data.error);
                          setErrorType(data.error_type)});
      }, []);
    console.log('errorMsg', errorMsg)

    return (
        <div className="App" style={{padding:"60px"}}> 
            <Typography gutterBottom variant="h3" align="center" >
            Data Auto-Augmentation
            </Typography>
            <Card style={{ maxWidth: 900, padding: "10px 5px", margin: "0 auto" }}>
                <CardContent>
                <Grid style={{padding:"40px"}}>
                    <Typography gutterBottom variant='h5'>
                        Oops! We found an error!
                    </Typography>

                    <Grid style={{padding:"10px"}}>
                        <Typography gutterBottom variant='body1'>{errorMsg}</Typography>
                    </Grid>

                    {errorType==="incorret dataset" &&
                    <Grid>
                        <Typography>
                            Uploaded dataset folder should have the following strcture:
                        </Typography>
                        <Grid
                            style={{
                            display: "flex",
                            alignItem: "center",
                            justifyContent: "center",
                            padding:"30px"
                            }}
                        >
                        <CardMedia
                            style={{
                                width: "auto",
                                maxHeight: "300px"
                            }}
                            component="img"
                            image={datafolder}
                            title="Contemplative Reptile"
                            />
                        </Grid>
                    </Grid>
                    }

                    {errorType==='not a zip file' && 
                        <Grid>
                        <Typography>
                            Uploaded dataset folder should have the following strcture:
                        </Typography>
                        
                        <Grid
                            style={{
                            display: "flex",
                            alignItem: "center",
                            justifyContent: "center",
                            padding:"30px"
                            }}
                        >
                        <CardMedia
                            style={{
                                width: "auto",
                                maxHeight: "300px"
                            }}
                            component="img"
                            image={datafolder}
                            />
                        </Grid>

                        </Grid>
                            }

                    <Button variant="contained"
                        size='large'
                        as={Link}
                        to="/"
                        >
                            Home
                    </Button>
                </Grid>

                </CardContent>
            </Card>
        </div>
    )
}