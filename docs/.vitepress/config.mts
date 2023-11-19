import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Ai00 server",
  description: "Awesome AI",
  base: '/ai00_rwkv_server/',
  lastUpdated: true,
  cleanUrls: true,
  appearance:'dark',
  markdown: {
    math: true
  },
 
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    
   socialLinks: [
      { icon: 'github', link: 'https://github.com/cgisky1980/ai00_rwkv_server' }
    ],



 
 

  },
  locales: {
    root: {
      label: '简体中文',
      lang: 'zh-CN', // optional, will be added  as `lang` attribute on `html` tag
      themeConfig: {
        nav: [
          { text: '首页', link: '/' },
          { text: '指南', link: '/markdown-examples' }
        ],
        lastUpdated: {
          text: '最后更新于',
          formatOptions: {
            dateStyle: 'full',
            timeStyle: 'medium'
          }
        },
        editLink: {
          pattern: 'https://github.com/cgisky1980/ai00_rwkv_server/edit/main/docs/:path',
          text: '在GITHUB编辑此页面'
        },
        sidebar: [
          {
            text: '例子',
            items: [
              { text: 'Markdown Examples', link: '/markdown-examples' },
              { text: 'Runtime API Examples', link: '/api-examples' }
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
