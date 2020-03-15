<template>
  <div class="container">
    <div class="row">
      <div id="col-fix" class="col-sm-10">
        <h1 id="title">WhichBee</h1>
        <hr><br><br>
        <alert :message=message v-if="showMessage"></alert>
        <div style="text-align:center;">
          <button type="button"
                  id="upload_btn"
                  class="btn btn-success btn-sm"
                  v-b-modal.upload-modal>
              Upload
          </button>
        </div>
        <br><br>
        <!-- <table class="table table-hover">
          <thead>
            <tr>
              <th scope="col">Name</th>
              <th scope="col">Path</th>
              <th scope="col">Time</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(image, index) in images" :key="index">
              <td>{{ image.name }}</td>
              <td>{{ image.path }}</td>
              <td>{{ image.time }}</td>
              <td>
                <div class="btn-group" role="group">
                  <button type="button" class="btn btn-warning btn-sm">Update</button>
                  <button type="button" class="btn btn-danger btn-sm">Delete</button>
                </div>
              </td>
            </tr>
          </tbody>
        </table> -->
      </div>
    </div>
    <b-modal ref="addUploadModal"
             id="upload-modal"
             title="Upload Images"
             hide-footer>
        <b-form @submit="onSubmit" @reset="onReset" class="w-100">
            <b-form-group id="form-title-group"
                          label="Images:"
                          label-for="form-image-input">
                <textarea rows="10"
                          disabled
                          style="width:100%;"
                          v-model="imagesText">
                </textarea>
            </b-form-group>
            <input type="file"
                   id="image_upload"
                   name="file[]"
                   ref="fileInput"
                   style="display:none"
                   multiple
                   @change="update($event)"/>
            <b-button id="browse_btn" @click="trigger">Browse</b-button>
            <b-button type="submit" variant="primary" class="submission_btn">Submit</b-button>
            <b-button type="reset" variant="danger" class="submission_btn">Reset</b-button>
        </b-form>
    </b-modal>
    <grid></grid>
  </div>
</template>

<script>
import axios from 'axios';
import Alert from './Alert.vue';
import Grid from './Grid.vue';

export default {
  data() {
    return {
      images: [],
      attachments: [],
      files: new FormData(),
      tempFiles: {},
      message: '',
      imagesText: '',
      showMessage: false,
    };
  },
  components: {
    alert: Alert,
    grid: Grid,
  },
  methods: {
    trigger() {
      this.$refs.fileInput.click();
    },
    update(evt) {
      for (let i = 0; i < evt.target.files.length; i += 1) {
        const file = evt.target.files[i];
        this.attachments.push(file);
        const temp = 'files['.concat(i).concat(']');
        this.files.append(temp, file);
      }
      this.tempFiles = evt.target.files;
      let strImage = '';
      for (let i = 0; i < this.tempFiles.length; i += 1) {
        const file = this.tempFiles[i];
        strImage = strImage.concat(file.name.toString().concat('\n'));
      }
      this.imagesText = strImage;
    },
    getImages() {
      const path = 'http://localhost:5000/images';
      axios.get(path)
        .then((res) => {
          this.images = res.data.images;
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.error(error);
        });
    },
    addImage(payload) {
      const path = 'http://localhost:5000/images';
      const config = {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      };
      axios.post(path, payload, config)
        .then(() => {
          this.getImages();
          this.message = 'Image added!';
          this.showMessage = true;
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.log(error);
          this.getImages();
        });
    },
    initForm() {
      this.files = new FormData();
      this.tempFiles = {};
      this.imagesText = '';
      this.attachments = [];
    },
    onSubmit(evt) {
      evt.preventDefault();
      this.$refs.addUploadModal.hide();
      for (let i = 0; i < this.attachments.length; i += 1) {
        const temp = 'files['.concat(i).concat(']');
        this.files.append(temp, this.attachments[i]);
      }
      const payload = this.files;
      this.addImage(payload);
      this.initForm();
    },
    onReset(evt) {
      evt.preventDefault();
      this.$refs.addUploadModal.hide();
      this.initForm();
    },
  },
  created() {
    this.getImages();
  },
};
</script>


<style>
#col-fix {
  max-width: 100%;
  flex: 100%;
}

#title {
  text-align: center;
  font-size: 5em;
}

#upload_btn {
  font-size: 3em;
  border-radius: 0.5em;
}

.submission_btn {
  float: right;
  margin-left: 0.5em;
}
</style>
