import Vue from 'vue';
import VueRouter from 'vue-router';
import Books from '../components/Books.vue';
import Main from '../components/Main.vue';

Vue.use(VueRouter);

const routes = [
  {
    path: '/',
    name: 'Main',
    component: Main,
  },
  {
    path: '/books',
    name: 'Books',
    component: Books,
  },
];

const router = new VueRouter({
  mode: 'history',
  base: process.env.BASE_URL,
  routes,
});

export default router;
