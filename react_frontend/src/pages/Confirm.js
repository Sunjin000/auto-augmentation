import React, { useState, useEffect } from "react";
import { Grid, ListItem, ListItemAvatar, ListItemText, Card, CardContent, Typography, Button } from '@mui/material';
import CheckCircleOutlineRoundedIcon from '@mui/icons-material/CheckCircleOutlineRounded';
import TuneRoundedIcon from '@mui/icons-material/TuneRounded';
import {useNavigate, Route} from "react-router-dom";

export default function Confirm() {
    const [myData, setMyData] = useState([])
    const [dataset, setDataset] = useState()
    const [network, setNetwork] = useState()
    const [yes, setYes] = useState()

    console.log('already in confirm react')
  useEffect(() => {
    const res = fetch('/home').then(
      response => response.json()
      ).then(data => {setMyData(data);
        if (data.ds == 'Other'){setDataset(data.ds_name)} else {setDataset(data.ds)};
        if (data.IsLeNet == 'Other'){setNetwork(data.network_name)} else {setNetwork(data.IsLeNet)};
        setYes('hey');
        console.log('setYes', yes);
    });
  }, []);


  let navigate = useNavigate();
  const onSubmit = async () => {
    navigate('/progress', {replace:true});
  };

    return (
        <div className="App" style={{padding:"60px"}}>
            <Typography gutterBottom variant="h3" align="center" >
            Data Auto-Augmentation {yes}
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
                                <ListItemText primary="Batch size" secondary={myData.batch_size} />
                            </ListItem>
                        </Grid>
                        <Grid xs={12} sm={6} item > 
                            <ListItem>
                                <ListItemAvatar>
                                    <CheckCircleOutlineRoundedIcon color="primary" fontSize='large'/>
                                </ListItemAvatar>
                                <ListItemText primary="Dataset selection" secondary={dataset} />
                            </ListItem>
                        </Grid>
                        <Grid xs={12} sm={6} item> 
                            <ListItem>
                                <ListItemAvatar>
                                    <TuneRoundedIcon color="primary" fontSize='large'/>
                                </ListItemAvatar>
                                <ListItemText primary="Learning rate" secondary={myData.learning_rate} />
                            </ListItem>
                        </Grid>
                        <Grid xs={12} sm={6} item> 
                            <ListItem>
                                <ListItemAvatar>
                                    <CheckCircleOutlineRoundedIcon color="primary" fontSize='large'/>
                                </ListItemAvatar>
                                <ListItemText primary="Network selection" secondary={network} />
                            </ListItem>
                        </Grid>
                        <Grid xs={12} sm={6} item> 
                            <ListItem>
                                <ListItemAvatar>
                                    <TuneRoundedIcon color="primary" fontSize='large'/>
                                </ListItemAvatar>
                                <ListItemText primary="Dataset Proportion" secondary={myData.toy_size} />
                            </ListItem>
                        </Grid>
                        <Grid xs={12} sm={6} item> 
                            <ListItem>
                                <ListItemAvatar>
                                    <CheckCircleOutlineRoundedIcon color="primary" fontSize='large'/>
                                </ListItemAvatar>
                                <ListItemText primary="Auto-augment learner selection" secondary={myData.auto_aug_learner} />
                            </ListItem>
                        </Grid>
                        <Grid xs={12} sm={6} item> 
                            <ListItem>
                                <ListItemAvatar>
                                    <TuneRoundedIcon color="primary" fontSize='large'/>
                                </ListItemAvatar>
                                <ListItemText primary="Iterations" secondary={myData.iterations} />
                            </ListItem>
                        </Grid>
                        </Grid>
                        </Grid>
                    
                        <Button
                            type="submit"
                            variant="contained"
                            color='success'
                            size='large'
                            onClick={onSubmit}
                        >
                            Confirm
                        </Button>

                    </CardContent>
                </Card>
            </Grid>
        </div>
    )
}