import React, { useState, useEffect } from "react";
import { Grid, List, ListItem, Avatar, ListItemAvatar, ListItemText, Card, CardContent, Typography, Button, CardMedia } from '@mui/material';
import output from './pytest.png'
import {useNavigate, Route} from "react-router-dom";

export default function Result() {

    
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
                            <img src={output} alt='output' />
                        </Grid>
                        <Grid xs={5} item> 
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
                        >
                            Download
                    </Button>
                    <Typography>
                        Please follow our documentation to apply this policy to your dataset.
                    </Typography>
                </CardContent>
            </Card>

        </div>
    )
}