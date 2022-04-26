import React, { useState, useEffect } from "react";
import { Grid, RadioGroup, FormControlLabel, FormControl, FormLabel, Radio, Card, CardContent, Typography, AlertTitle } from '@mui/material';
import {Button, TextField, Checkbox, Alert} from '@mui/material';
import { useForm, Controller} from "react-hook-form";
import SendIcon from '@mui/icons-material/Send';
import { CardActions, Collapse, IconButton } from "@mui/material";
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { styled } from '@mui/material/styles';
import {useNavigate, Route} from "react-router-dom";




const ExpandMore = styled((props) => {
    const { expand, ...other } = props;
    return <IconButton {...other} />;
  })(({ theme, expand }) => ({
    transform: !expand ? 'rotate(0deg)' : 'rotate(180deg)',
    marginLeft: 'auto',
    transition: theme.transitions.create('transform', {
      duration: theme.transitions.duration.shortest,
    }),
  }));




export default function Home() {
    const [selectLearner, setSelectLearner] = useState([]);
    const [selectAction, setSelectAction] = useState([]);

    // for form submission  
    const {register, control, handleSubmit, setValue, watch, formState: { errors, dirtyFields}} = useForm();
    const watchFileds = watch(['select_dataset', 'select_network']);

    let navigate = useNavigate();

    const onSubmit = async (data) => {
        console.log('data', data);

        const formData = new FormData();

        formData.append("ds_upload", data.ds_upload[0]);
        formData.append("network_upload", data.network_upload[0]);
        formData.append("batch_size", data.batch_size)
        formData.append("toy_size", data.toy_size)
        formData.append("iterations", data.iterations)
        formData.append("learning_rate", data.learning_rate)
        formData.append("select_action", data.select_action)
        formData.append("select_dataset", data.select_dataset)
        formData.append("select_learner", data.select_learner)
        formData.append("select_network", data.select_network)

        console.log('>>> this is the user input in formData')
        for (var key of formData.entries()) {
            console.log(key[0] + ', ' + key[1])}
        
        var responseClone
        const res = await fetch('/home', {
        method: 'POST',
        body: formData
        }).then((response) => response.json());
        
        navigate('/confirm', {replace:true});
        // 
        ///////// testing
        // .then((response)=> {
        //     responseClone = response.clone(); // 2
        //     return response.json();
        // })
        // .then(function (data) {
        //     console.log('data from flask', data)
        // }, function (rejectionReason) { // 3
        //     console.log('Error parsing JSON from response:', rejectionReason, responseClone); // 4
        //     responseClone.text() // 5
        //     .then(function (bodyText) {
        //         console.log('Received the following instead of valid JSON:', bodyText); // 6
        //     });
        // });
        
    };

    
    // body: JSON.stringify(data)
    // console.log('errors', errors); 
    // console.log('handleSubmit', handleSubmit)

    
    // handling action selection
    const handleActionSelect = (value) => {
        const isPresent = selectAction.indexOf(value);
        if (isPresent !== -1) {
        const remaining = selectAction.filter((item) => item !== value);
        setSelectAction(remaining);
        } else {
        setSelectAction((prevItems) => [...prevItems, value]);
        }
    };

    useEffect(() => {
        setValue('select_action', selectAction); 
    }, [selectAction]);

    // collpase
    const [expanded, setExpanded] = React.useState(false);

    const handleExpandClick = () => {
        setExpanded(!expanded);
    };

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

                            <FormControl style={{ maxWidth: 800, padding:"20px"}} error={Boolean(errors.select_dataset)}>
                                <FormLabel id="select_dataset" align="left" variant="h6">
                                    Please select the dataset you'd like to use here or select 'Other' if you would like to upload your own dataset
                                </FormLabel>
                                <Controller 
                                        name='select_dataset'
                                        control={control}
                                        rules={{ required: true }}
                                        render={({field: { onChange, value }}) => (
                                    <RadioGroup
                                        row
                                        aria-labelledby="select_dataset"
                                        // defaultValue="Other"
                                        name="select_dataset"
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
                                    </RadioGroup> 
                                    )}
                                />
                                {errors.select_dataset && errors.select_dataset.type === "required" && 
                                    <Alert severity="error">
                                        <AlertTitle>This field is required</AlertTitle>
                                    </Alert>}
                                <Button
                                variant="contained"
                                component="label"
                                >
                                Upload File
                                <input
                                    {...register('ds_upload')}
                                    name="ds_upload"
                                    type="file"
                                    hidden
                                />
                                </Button>
                                {dirtyFields.ds_upload && <Alert severity="success" variant='outlined'>File Submitted</Alert>}
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

                            <FormControl style={{ maxWidth: 800, padding:"20px"}} error={Boolean(errors.select_network)}>
                                <FormLabel id="select_network" align="left" variant="h6">
                                    Please select the network you'd like to use here or select 'Other' if you would like to upload your own network
                                </FormLabel>
                                <Controller 
                                        name='select_network'
                                        control={control}
                                        rules={{ required: true }}
                                        render={({field: { onChange, value }}) => (
                                    <RadioGroup
                                        row
                                        aria-labelledby="select_network"
                                        name="select_network"
                                        align="centre"
                                        value={value ?? ""} 
                                        onChange={onChange}
                                        >
                                        <FormControlLabel value="LeNet" control={<Radio />} label="LeNet" />
                                        <FormControlLabel value="SimpleNet" control={<Radio />} label="SimpleNet" />
                                        <FormControlLabel value="EasyNet" control={<Radio />} label="EasyNet" />
                                        <FormControlLabel value="Other" control={<Radio />} label="Other" /> 
                                    </RadioGroup> )}
                                />
                                {errors.select_network && errors.select_network.type === "required" && 
                                    <Alert severity="error">
                                        <AlertTitle>This field is required</AlertTitle>
                                    </Alert>}
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
                                    {...register('network_upload')}
                                    name="network_upload"
                                    type="file"
                                    hidden
                                />
                                </Button>
                                {dirtyFields.network_upload && <Alert severity="success" variant='outlined'>File Submitted</Alert>}
                            </FormControl>
                        </CardContent>
                    </Card>
                </Grid>

                <Grid style={{padding:"30px 0px"}}>
                    <Card style={{ maxWidth: 900, padding: "10px 5px", margin: "0 auto" }}>
                        <CardContent>
                            <Typography gutterBottom variant="h5" align="left">
                                Auto-augment Learner Selection
                            </Typography> 

                            <FormControl style={{ maxWidth: 800, padding:"20px"}} error={Boolean(errors.select_learner)}>
                                <FormLabel id="select_learner" align="left" variant="h6">
                                    Please select the auto-augment learners you'd like to use (multiple learners can be selected)
                                </FormLabel>
                                <Controller 
                                        name='select_learner'
                                        control={control}
                                        rules={{ required: true }}
                                        render={({field: { onChange, value }}) => (
                                    <RadioGroup
                                        row
                                        aria-labelledby="select_learner"
                                        name="select_learner"
                                        align="centre"
                                        value={value ?? ""} 
                                        onChange={onChange}
                                        >
                                        <FormControlLabel value="UCB learner" control={<Radio />} label="UCB learner" />
                                        <FormControlLabel value="Evolutionary learner" control={<Radio />} label="Evolutionary learner" />
                                        <FormControlLabel value="Random Searcher" control={<Radio />} label="Random Searcher" />
                                        <FormControlLabel value="GRU Learner" control={<Radio />} label="GRU Learner" /> 
                                    </RadioGroup> )}
                                />
                                {errors.select_learner && errors.select_learner.type === "required" && 
                                    <Alert severity="error">
                                        <AlertTitle>This field is required</AlertTitle>
                                    </Alert>}
                                <Typography style={{ maxWidth: 800}} variant="body2" color="textSecondary" component="p" gutterBottom align="left">
                                    (give user some recommendation here...)
                                </Typography>
                            </FormControl>
                        </CardContent>
                    </Card>
                </Grid>


                <Grid style={{padding:"30px 0px", maxWidth: 900, margin: "0 auto"}}>
                    <Card style={{ maxWidth: 900, padding: "10px 5px", margin: "0 auto" }}>
                        <CardContent>
                            <Typography variant="h5" align="left">
                                Advanced Search
                            </Typography>
                        </CardContent>
                        <CardActions disableSpacing>
                            <ExpandMore
                            expand={expanded}
                            onClick={handleExpandClick}
                            aria-expanded={expanded}
                            aria-label="show more"
                            >
                            <ExpandMoreIcon />
                            </ExpandMore>
                        </CardActions>
                        <Collapse in={expanded} timeout="auto" unmountOnExit>
                            <Grid container
                                    spacing={0}
                                    direction="column"
                                    alignItems="center"
                                    justify="center">
                            <CardContent>
                            <Typography gutterBottom variant="subtitle1" align='left'>
                                    Please input the hyperparameter you'd like to train your dataset with
                            </Typography>
                            <Grid container spacing={1} style={{maxWidth:800, padding:"10px 10px"}}>
                                <Grid xs={12} sm={6} item>
                                    <TextField type="number" InputProps={{ inputProps: { min: 0} }} {...register("batch_size")} name="batch_size" placeholder="Batch Size" label="Batch Size" variant="outlined" fullWidth />
                                </Grid>
                                <Grid xs={12} sm={6} item>
                                    <TextField type="number" inputProps={{step: "0.000000001",min: 0}} {...register("learning_rate")} name="learning_rate" placeholder="Learning Rate" label="Learning Rate" variant="outlined" fullWidth />
                                </Grid>
                                <Grid xs={12} sm={6} item>
                                    <TextField type="number" InputProps={{ inputProps: { min: 0} }} {...register("iterations")} name="iterations" placeholder="Number of Iterations" label="Iterations" variant="outlined" fullWidth />
                                </Grid>
                                <Grid xs={12} sm={6} item>
                                    <TextField type="number" inputProps={{step: "0.01", min: 0}} {...register("toy_size")} name="toy_size" placeholder="Dataset Proportion" label="Dataset Proportion" variant="outlined" fullWidth />
                                </Grid>
                                <FormLabel variant="h8" align='centre'>
                                    * Dataset Proportion defines the percentage of original dataset our auto-augment learner will use to find the 
                                    augmentation policy. If your dataset is large, we recommend you to set Dataset Proportion a small value (0.1-0.3). 
                                </FormLabel>
                            </Grid>

                            <Grid style={{maxWidth:800, padding:"40px 10px"}}>
                                <Typography gutterBottom variant="subtitle1" align='left'>
                                    Please select augmentation methods you'd like to exclude 
                                </Typography>
                                <div>
                                    {['ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 'Brightness',
                                    'Color', 'Contrast', 'Sharpness', 'Posterize', 'Solarize', 'AutoContrast', 
                                    'Equalize', 'Invert'].map((option) => {
                                    return (
                                        <FormControlLabel
                                        control={
                                            <Controller
                                            name='select_action'
                                            render={({}) => {
                                                return (
                                                <Checkbox
                                                    checked={selectAction.includes(option)}
                                                    onChange={() => handleActionSelect(option)}/> );
                                            }}
                                            control={control}
                                            />}
                                        label={option}
                                        key={option}
                                        />
                                    );
                                    })}
                                </div>
                            </Grid>
                            </CardContent>
                            </Grid>
                        </Collapse>
                         
                    </Card>
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
                {watchFileds[0]==='Other' && !dirtyFields.ds_upload && 
                    <Alert severity="error" variant='standard'>Please upload your dataset 
                    zip file or select one of the dataset we have provided</Alert>}
                {watchFileds[1]==='Other' && !dirtyFields.network_upload && 
                    <Alert severity="error" variant='standard'>Please upload your network 
                    .pkl file or select one of the network we have provided</Alert>}
                </form>
                        
            </Grid>
        </div>
            
        );
    }

