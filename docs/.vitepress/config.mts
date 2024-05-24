import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Ai00 server",
  description: "Awesome AI",
  base: '/ai00_server/',
  lastUpdated: true,
  cleanUrls: true,
  appearance:'dark',
  markdown: {
    math: true
  },
 
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    
   socialLinks: [
      { icon: 'github', link: 'https://github.com/Ai00-X/ai00_server' }
    ],



 
 

  },
  locales: {
    root: {
      label: '简体中文',
      lang: 'zh-CN', // optional, will be added  as `lang` attribute on `html` tag
      themeConfig: {
        nav: [
          { text: '首页', link: '/' },
          { text: '快速上手', link: '/Simple-Usage' }
        ],
        lastUpdated: {
          text: '最后更新于',
          formatOptions: {
            dateStyle: 'full',
            timeStyle: 'medium'
          }
        },
        editLink: {
          pattern: 'https://github.com/Ai00-X/ai00_server/edit/main/docs/:path',
          text: '在GITHUB编辑此页面'
        },
        sidebar: [
          {
            text: 'Ai00 文档',
            items: [
              { text: '了解 Ai00', link: '/Introduction' },
              { text: '快速上手', link: '/Simple-Usage' },
              { text: '进阶功能', link: '/Ai00-Features' },
              { text: '常见问题', link: '/FAQ' }
            ]
          } 
        ],

      },

    },
    en: {
      label: 'English',
      lang: 'en', // optional, will be added  as `lang` attribute on `html` tag
      link: '/en/' // default /fr/ -- shows on navbar translations menu, can be external

      // other locale specific properties...
    }
  }
 

})
