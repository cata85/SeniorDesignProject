<template>
  <div class="container">
    <div class="row">
      <div id="col-fix" class="col-sm-10">
        <h1 id="title">WhichBee</h1>
        <hr>
        <h4 id="instructions">
          Upload photo of a Bumblebee to get it's species.
          (Make sure the wings are in clear view)
        </h4>
        <br><br>
        <div v-if="showMessage">
          <!-- <img :src="message" /> -->
          <alert :imgSrc="imgSrc" :imgName="imgName" :imgId="imgId"></alert>
        </div>
        <!-- <img v-if="showMessage" :src="message" /> -->
        <div style="text-align:center;">
          <button type="button"
                  id="upload_btn"
                  class="btn btn-success btn-sm"
                  v-b-modal.upload-modal>
              Upload
          </button>
        </div>
        <br><br>
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
      imgSrc: '',
      imagesText: '',
      showMessage: false,
      imgId: '',
      imgName: '',
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
        encoding: 'binary',
        responseType: 'arraybuffer',
      };
      axios.post(path, payload, config)
        .then((res) => {
          this.getImages();
          const img = 'data:image/jpeg;base64,'.concat(Buffer.from(res.data, 'utf-8').toString('base64'));
          const tempName = res.headers['x-suggested-filename'];
          this.imgName = tempName.replace('.jpg', '');
          this.imgId = tempName.replace('.jpg', '');
          this.imgSrc = img;
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

#instructions {
  text-align: center;
}

#upload_btn {
  font-size: 3em;
  border-radius: 0.5em;
}

.submission_btn {
  float: right;
  margin-left: 0.5em;
}

body {
  background: rgb(251,251,63);
  background: radial-gradient(circle, rgba(251,251,63,1) 0%, rgba(252,135,70,1) 100%);
}
</style>
