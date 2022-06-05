import "./App.css";
import * as React from "react";
import { useState } from "react";
import { styled } from "@mui/material/styles";
import Button from "@mui/material/Button";
import Stack from "@mui/material/Stack";
import AudioFileIcon from "@mui/icons-material/AudioFile";
import SendIcon from "@mui/icons-material/Send";
import Typography from "@mui/material/Typography";
import placeholder from "./assets/adrian-regeci-SAS0lq2QGLs-unsplash.jpg";
import axios from "axios";
import LoadingButton from "@mui/lab/LoadingButton";

const Input = styled("input")({
  display: "none",
});

function App() {
  const [selectedFile, setSelectedFile] = useState();
  const [loading, setLoading] = useState(false);
  const [genre, setGenre] = useState("");

  return (
    <>
      <Stack direction="row" backgroundColor="#222831" height="100vh">
        <Stack
          direction="column"
          alignItems="center"
          justifyContent="center"
          spacing={2}
          backgroundColor="#393E46"
          width="60%"
        >
          <Typography variant="h5" color="white">
            Music Classifier
          </Typography>
          {selectedFile && (
            <Typography variant="caption" color="white">
              Selected {selectedFile.name}
            </Typography>
          )}
          <label>
            <Input
              type="file"
              onChange={(event) => {
                const file = event.target.files[0];
                if (file) setSelectedFile(file);
              }}
            />
            <Button
              variant="contained"
              component="span"
              startIcon={<AudioFileIcon />}
            >
              Upload
            </Button>
          </label>
          <LoadingButton
            disabled={selectedFile ? false : true}
            loading={loading}
            loadingPosition="end"
            endIcon={<SendIcon />}
            variant="contained"
            onClick={() => {
              // post if have selected file
              if (selectedFile) {
                // `file` here for matching the files[`file`] on server side
                const formData = new FormData();
                formData.append("file", selectedFile);
                setLoading(true);
                axios
                  .post("/home", formData)
                  .then((res) => {
                    setLoading(false);
                    setGenre(res.data.text);
                  })
                  .catch((error) => {
                    console.log(error);
                  });
              }
            }}
          >
            {loading ? "Classifying" : "Classify"}
          </LoadingButton>

          {genre && (
            <Typography variant="h6" color="white">
              Genre {genre}
            </Typography>
          )}
        </Stack>
        <img src={placeholder} alt="music" />
      </Stack>
    </>
  );
}

export default App;
