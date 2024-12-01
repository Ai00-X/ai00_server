import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Ai00 server",
  description: "Awesome AI",
  base: '/ai00_server/',
  lastUpdated: true,
  ignoreDeadLinks: true,
  cleanUrls: true,
  appearance: 'dark',
  markdown: {
    math: true
  },

  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    search: {
      provider: 'local'
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/Ai00-X/ai00_server' }
    ],

  },
  locales: {
    root: {
      label: '简体中文',
      lang: 'zh-CN', // optional, will be added  as `lang` attribute on `html` tag
      themeConfig: {
        logo: '/logo.gif',
        nav: [
          { text: '指南', link: '/' },
          { text: '快速上手', link: '/doc-guide/quick-start' }
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
            text: '简介',
            collapsed: false,
            base: '/doc-guide/',
            items: [
              { text: '了解 Ai00', link: '/what-is-ai00' },
              {
                text: '快速上手',
                collapsed: false,
                items: [

                  { text: '下载编译包', link: '/release' },
                  { text: '从源码安装', link: '/source-install' },
                  { text: '配置文件', link: '/config' },

                ]
              },
            ]
          },

          {
            text: '模型',
            collapsed: false,
            base: '/doc-models/',
            items: [
              { text: '模型命名规范', link: '/models-name' },
              { text: 'RWKV基座模型', link: '/rwkv-base' },
              { text: 'LoRA', link: '/lora-model' },
              { text: '初始State', link: '/state-model' },
            ]
          }
          ,
          {
            text: 'WebUI',
            collapsed: false,
            base: '/doc-webui/',
            items: [
              { text: 'WebUI配置', link: '/webui-config' },
              { text: '对话例子', link: '/example-chat' },
              { text: '续写例子', link: '/example-write' },
              { text: '并发例子', link: '/example-batch' },
            ]
          }
          ,
          {
            text: 'API',
            collapsed: false,
            base: '/doc-api/',
            items: [
              { text: 'API接口列表', link: '/openai' },
              { text: '调试API', link: '/debug-api' },
              {
                text: 'SDK调用',
                base: 'sdk/',
                collapsed: false,
                items: [
                  { text: 'Python SDK', link: '/python-sdk' },
                  { text: 'JS SDK', link: '/js-sdk' },
                  { text: 'Rust SDK', link: '/rust-sdk' },
                ]
              },
            ]
          },
          {
            text: '其他',
            collapsed: false,
            base: '/doc-guide/',
            items: [
              { text: '进阶功能', link: '/features' },
              { text: '常见问题', link: '/FAQ' }
            ]
          },
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
