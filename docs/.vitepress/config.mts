import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Ai00 server",
  description: "Awesome AI",
  base: '/ai00_rwkv_server/',
  lastUpdated: true,
  cleanUrls: true,

  markdown: {
    math: true
  },
  
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    darkMode: true,

    nav: [
      { text: 'Home', link: '/' },
      { text: 'Examples', link: '/markdown-examples' }
    ],

    sidebar: [
      {
        text: 'Examples',
        items: [
          { text: 'Markdown Examples', link: '/markdown-examples' },
          { text: 'Runtime API Examples', link: '/api-examples' }
        ]
      },
      {
        text: 'Examples',
        items: [
          { text: 'Markdown Examples', link: '/markdown-examples' },
          { text: 'Runtime API Examples', link: '/api-examples' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/cgisky1980/ai00_rwkv_server' }
    ]
  },
  locales: {
    root: {
      label: '简体中文',
      lang: 'zh-CN', // optional, will be added  as `lang` attribute on `html` tag
    },
    en: {
      label: 'English',
      lang: 'en', // optional, will be added  as `lang` attribute on `html` tag
      link: '/en/' // default /fr/ -- shows on navbar translations menu, can be external

      // other locale specific properties...
    }
  }


})
