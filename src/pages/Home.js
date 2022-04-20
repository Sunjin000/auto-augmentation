import React, { useState, useEffect } from "react";
import { Grid, RadioGroup, FormControlLabel, FormControl, FormLabel, Radio, Card, CardContent, Typography } from '@mui/material';
import {Button, TextField, Checkbox, FormGroup, Box, Switch, Grow} from '@mui/material';
import { useForm, Controller} from "react-hook-form";
import SendIcon from '@mui/icons-material/Send';



const hyperparameter = (
    <Grid container spacing={1} style={{maxWidth:500, padding:"0px 10px"}}>
        <Grid xs={12} sm={6} item>
            <TextField name="batch_size" placeholder="Batch Size" label="Batch Size" variant="outlined" fullWidth />
        </Grid>
        <Grid xs={12} sm={6} item>
            <TextField name="learning_rate" placeholder="Learning Rate" label="Learning Rate" variant="outlined" fullWidth />
        </Grid>
        <Grid xs={12} sm={6} item>
            <TextField name="num_policies" placeholder="Number of Policies" label="Number of Policies" variant="outlined" fullWidth />
        </Grid>
        <Grid xs={12} sm={6} item>
            <TextField name="num_sub_policies" placeholder="Number of Subpolicies" label="Number of Subpolicies" variant="outlined" fullWidth />
        </Grid>
        <Grid xs={12} sm={6} item>
            <TextField name="iterations" placeholder="Number of Iterations" label="Iterations" variant="outlined" fullWidth />
        </Grid>
        <Grid xs={12} sm={6} item>
            <TextField name="toy_size" placeholder="Dataset Proportion" label="Dataset Proportion" variant="outlined" fullWidth />
        </Grid>

    </Grid>
);


const action_space = (
    <Grid style={{maxWidth:500, padding:"0px 10px"}}>
        <FormLabel id="select-action" align="left" variant="h6">
            Please select augmentation methods you'd like to exclude 
        </FormLabel>
        <FormGroup
        row
        aria-labelledby="select-action"
        name="select-action"
        >
            <FormControlLabel value="ShearX" control={<Checkbox />} label="ShearX" />
            <FormControlLabel value="ShearY" control={<Checkbox />} label="ShearY" />
            <FormControlLabel value="TranslateX" control={<Checkbox />} label="TranslateX" />
            <FormControlLabel value="TranslateY" control={<Checkbox />} label="TranslateY" />
            <FormControlLabel value="Rotate" control={<Checkbox />} label="Rotate" />
            <FormControlLabel value="Brightness" control={<Checkbox />} label="Brightness" />
            <FormControlLabel value="Color" control={<Checkbox />} label="Color" />
            <FormControlLabel value="Contrast" control={<Checkbox />} label="Contrast" />
            <FormControlLabel value="Sharpness" control={<Checkbox />} label="Sharpness" />
            <FormControlLabel value="Posterize" control={<Checkbox />} label="Posterize" />
            <FormControlLabel value="Solarize" control={<Checkbox />} label="Solarize" />
            <FormControlLabel value="AutoContrast" control={<Checkbox />} label="AutoContrast" />
            <FormControlLabel value="Equalize" control={<Checkbox />} label="Equalize" />
            <FormControlLabel value="Invert" control={<Checkbox />} label="Invert" />
        </FormGroup>
    </Grid>
)


// class Home extends React.Component{
export default function Home() {

    const [checked, setChecked] = useState(false); // advanced search toggle
    const [dsvalue, setDsvalue] = useState('Other'); // dataset selection
    const [netvalue, setNetvalue] = useState('Other'); // network selection

    const handleShow = () => {
      setChecked((prev) => !prev);
    };

    const handleDsChange = (event) => {
        setDsvalue(event.target.value);
    };

    const handleNetChange = (event) => {
        setNetvalue(event.target.value);
    };

// for form submission
    const { control, handleSubmit } = useForm();
    const onSubmit = data => console.log(data);

    
  

    // render(){
        return (
            
        <div className="App" style={{padding:"60px"}}> 
            <Typography gutterBottom variant="h3" align="center" >
            Data Auto-Augmentation 
            </Typography>
            <Grid >
                <form action="/home" method="POST" onSubmit={handleSubmit(onSubmit)}>
                <Grid style={{padding:"30px 0px"}}>
                    <Card style={{ maxWidth: 900, padding: "10px 5px", margin: "0 auto" }}>
                        <CardContent>
                            <Typography gutterBottom variant="h5" align="left">
                                Dataset Uploading
                            </Typography> 

                            <FormControl style={{ maxWidth: 800, padding:"20px"}}>
                                <FormLabel id="select-dataset" align="left" variant="h6">
                                    Please select the dataset you'd like to use here or select 'Other' if you would like to upload your own dataset
                                </FormLabel>
                                <Controller 
                                        name='select-dataset'
                                        control={control}
                                        rules={{ required: true }}
                                        render={({field: { onChange, value }}) => (
                                    <RadioGroup
                                        row
                                        aria-labelledby="select-dataset"
                                        // defaultValue="Other"
                                        name="select-dataset"
                                        align="centre"
                                        value={value ?? ""} 
                                        onChange={onChange}
                                        >
                                        <FormControlLabel value="MNIST" control={<Radio />} label="MNIST" />
                                        <FormControlLabel value="KMNIST" control={<Radio />} label="KMNIST" />
                                        <FormControlLabel value="FashionMNIST" control={<Radio />} label="FashionMNIST" />
                                        <FormControlLabel value="CIFAR10" control={<Radio />} label="CIFAR10" />
                                        <FormControlLabel value="CIFAR100" control={<Radio />} label="CIFAR100" />
                                        <FormControlLabel value="Other" control={<Radio />} label="Other" />
                                    </RadioGroup> )}
                                />
                                <Button
                                variant="contained"
                                component="label"

                                >
                                Upload File
                                <input
                                    type="file"
                                    hidden
                                />
                                </Button>
                            </FormControl>
                        </CardContent>
                    </Card>
                </Grid>
                <Grid style={{padding:"30px 0px"}}>
                    <Card style={{ maxWidth: 900, padding: "10px 5px", margin: "0 auto" }}>
                        <CardContent>
                            <Typography gutterBottom variant="h5" align="left">
                                Network Uploading
                            </Typography> 

                            <FormControl style={{ maxWidth: 800, padding:"20px"}}>
                                <FormLabel id="select-network" align="left" variant="h6">
                                    Please select the network you'd like to use here or select 'Other' if you would like to upload your own network
                                </FormLabel>
                                <RadioGroup
                                    row
                                    aria-labelledby="select-network"
                                    defaultValue="Other"
                                    name="select-network"
                                    align="centre"
                                    value={dsvalue}
                                    onChange={handleDsChange}
                                    >
                                    <FormControlLabel value="LeNet" control={<Radio />} label="LeNet" />
                                    <FormControlLabel value="SimpleNet" control={<Radio />} label="SimpleNet" />
                                    <FormControlLabel value="EasyNet" control={<Radio />} label="EasyNet" />
                                    <FormControlLabel value="Other" control={<Radio />} label="Other" /> 
                                </RadioGroup>
                                <Typography style={{ maxWidth: 750}} variant="body2" color="textSecondary" component="p" gutterBottom align="left">
                                    The networks provided above are for demonstration purposes. The relative training time is: LeNet {'>'} SimpleNet {'>'} EasyNet. 
                                    We recommend you to choose EasyNet for a quick demonstration of how well our auto-augment agents can perform. 
                                </Typography>
                                <Button
                                variant="contained"
                                component="label"
                                >
                                Upload File
                                <input
                                    type="file"
                                    hidden
                                />
                                </Button>
                            </FormControl>
                        </CardContent>
                    </Card>
                </Grid>

                <Grid style={{padding:"30px 0px"}}>
                    <Card style={{ maxWidth: 900, padding: "10px 5px", margin: "0 auto" }}>
                        <CardContent>
                            <Typography gutterBottom variant="h5" align="left">
                                Auto-augment Agent Selection
                            </Typography> 

                            <FormControl style={{ maxWidth: 800, padding:"20px"}}>
                                <FormLabel id="select-learner" align="left" variant="h6">
                                    Please select the auto-augment learner you'd like to use
                                </FormLabel>
                                <RadioGroup
                                    row
                                    aria-labelledby="select-learner"
                                    defaultValue="UCB"
                                    name="select-learner"
                                    align="centre"
                                    value={netvalue}
                                    onChange={handleNetChange}
                                    >
                                    <FormControlLabel value="UCB" control={<Radio />} label="UCB" />
                                    <FormControlLabel value="Evolutionary" control={<Radio />} label="Evolutionary Learner" />
                                    <FormControlLabel value="Random Searcher" control={<Radio />} label="Random Searcher" />
                                    <FormControlLabel value="GRU Learner" control={<Radio />} label="GRU Learner" /> 
                                </RadioGroup>
                                <Typography style={{ maxWidth: 800}} variant="body2" color="textSecondary" component="p" gutterBottom align="left">
                                    (give user some recommendation here...)
                                </Typography>
                            </FormControl>
                        </CardContent>
                    </Card>
                </Grid>


                <Grid style={{padding:"30px 0px", maxWidth: 900, margin: "0 auto"}}>
                            <Typography gutterBottom variant="h5" align="left">
                                Advanced Search
                            </Typography>

                            <Box sx={{ maxHeight: 500 }}>
                                <FormControlLabel
                                    control={<Switch checked={checked} onChange={handleShow} />}
                                    label="Show"
                                />
                                <Box sx={{ display: 'flex' }}>
                                    <Grow in={checked}>{hyperparameter}</Grow>
                                    {/* Conditionally applies the timeout prop to change the entry speed. */}
                                    <Grow
                                    in={checked}
                                    style={{ transformOrigin: '0 0 0' }}
                                    {...(checked ? { timeout: 1000 } : {})}
                                    >
                                    {action_space}
                                    </Grow>
                                </Box>
                            </Box>
                </Grid>
                <Button
                    type="submit"
                    variant="contained"
                    color='success'
                    size='large'
                    endIcon={<SendIcon />}
                >
                    Submit Form
                </Button>
                </form>
                        
            </Grid>
        </div>
            
        );
    }

// export default Home;