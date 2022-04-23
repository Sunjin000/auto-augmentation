import React, { useState, useEffect } from "react";
import { Grid, List, ListItem, Avatar, ListItemAvatar, ListItemText, Card, CardContent, Typography, Button, TextField } from '@mui/material';
import CheckCircleOutlineRoundedIcon from '@mui/icons-material/CheckCircleOutlineRounded';
import TuneRoundedIcon from '@mui/icons-material/TuneRounded';

export default function Training() {
    const [myData, setMyData] = useState([{}])
  useEffect(() => {
    fetch('/api').then(
      response => {console.log('response', response); response.json()}
    ).then(data => {console.log('training', data); 
        })
  }, []);


    return (
        <div className="App" style={{padding:"60px"}}>
            <Typography gutterBottom variant="h3" align="center" >
            Data Auto-Augmentation
            </Typography>
            <Grid>
                <Card style={{ maxWidth: 900, padding: "10px 5px", margin: "0 auto" }}>
                    <CardContent>
                        <Typography gutterBottom variant="h6" align="left">
                            Please confirm the following information:
                        </Typography> 
                        <Grid alignItems="center" justify="center" >
                        <Grid style={{maxWidth: 700, padding: "20px 20px"}} container spacing={4} alignItems="center" >
                        <Grid xs={12} sm={6} item> 
                            <ListItem>
                                <ListItemAvatar>
                                    <TuneRoundedIcon color="primary" fontSize='large'/>
                                </ListItemAvatar>
                                <ListItemText primary="Batch size" secondary="[Batch size]" />
                            </ListItem>
                        </Grid>
                        <Grid xs={12} sm={6} item > 
                            <ListItem>
                                <ListItemAvatar>
                                    <CheckCircleOutlineRoundedIcon color="primary" fontSize='large'/>
                                </ListItemAvatar>
                                <ListItemText primary="Dataset selection" secondary="[Dataset]" />
                            </ListItem>
                        </Grid>
                        <Grid xs={12} sm={6} item> 
                            <ListItem>
                                <ListItemAvatar>
                                    <TuneRoundedIcon color="primary" fontSize='large'/>
                                </ListItemAvatar>
                                <ListItemText primary="Learning rate" secondary="[Learning rate]" />
                            </ListItem>
                        </Grid>
                        <Grid xs={12} sm={6} item> 
                            <ListItem>
                                <ListItemAvatar>
                                    <CheckCircleOutlineRoundedIcon color="primary" fontSize='large'/>
                                </ListItemAvatar>
                                <ListItemText primary="Network selection" secondary="[Network selection]" />
                            </ListItem>
                        </Grid>
                        <Grid xs={12} sm={6} item> 
                            <ListItem>
                                <ListItemAvatar>
                                    <TuneRoundedIcon color="primary" fontSize='large'/>
                                </ListItemAvatar>
                                <ListItemText primary="Dataset Proportion" secondary="[Dataset Proportion]" />
                            </ListItem>
                        </Grid>
                        <Grid xs={12} sm={6} item> 
                            <ListItem>
                                <ListItemAvatar>
                                    <CheckCircleOutlineRoundedIcon color="primary" fontSize='large'/>
                                </ListItemAvatar>
                                <ListItemText primary="Auto-augment learner selection" secondary="[Auto-augment learner selection]" />
                            </ListItem>
                        </Grid>
                        <Grid xs={12} sm={6} item> 
                            <ListItem>
                                <ListItemAvatar>
                                    <TuneRoundedIcon color="primary" fontSize='large'/>
                                </ListItemAvatar>
                                <ListItemText primary="Iterations" secondary="[Iterations]" />
                            </ListItem>
                        </Grid>
                        </Grid>
                        </Grid>
                    
                        <Button
                            type="submit"
                            variant="contained"
                            color='success'
                            size='large'
                        >
                            Confirm
                        </Button>

                    </CardContent>
                </Card>
            </Grid>
        </div>
    )
}