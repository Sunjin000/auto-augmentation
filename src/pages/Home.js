import React from 'react';

class Home extends React.Component{
    render(){
        return (
            <div>
                <h1>Meta Reinforcement Learning for Data Augmentation</h1>
                <h3>Choose your dataset</h3>
                <form action="/user_input" method="POST" enctype="multipart/form-data">
                    <label for="dataset_upload">You can upload your dataset folder here:</label>
                    <input type="file" id='dataset_upload' name="dataset_upload" class="upload" /><br></br>
                </form>
            </div>
            
        );
    }
}

export default Home;