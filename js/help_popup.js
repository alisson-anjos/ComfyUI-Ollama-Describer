import { app } from "../../scripts/app.js";

const categories = ["Ollama/Agent", "Ollama/Tools", "Ollama"];

app.registerExtension({
	name: "OllamaDescriber.HelpPopup",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		try {
			categories.forEach(category => {
                if (nodeData?.category?.startsWith(category)) {
                    addDocumentation(nodeData, nodeType);
                }
            });
		} catch (error) {
			console.error("Error in registering OllamaDescriber.HelpPopup", error);
		}
	},
});

const create_documentation_stylesheet = () => {
    const tag = 'ollama-documentation-stylesheet'

    let styleTag = document.head.querySelector(tag)

    if (!styleTag) {
      styleTag = document.createElement('style')
      styleTag.type = 'text/css'
      styleTag.id = tag
      styleTag.innerHTML = `
      .ollama-documentation-popup {
        background: var(--comfy-menu-bg);
        position: absolute;
        color: var(--fg-color);
        font: 12px monospace;
        line-height: 1.5em;
        padding: 10px;
        border-radius: 10px;
        border-style: solid;
        border-width: medium;
        border-color: var(--border-color);
        z-index: 5;
        overflow: hidden;
       }
       .content-wrapper {
        overflow: auto;
        max-height: 100%;
        width: 300px; /* Default size */
        &::-webkit-scrollbar {
           width: 6px;
        }
        &::-webkit-scrollbar-track {
           background: var(--bg-color);
        }
        &::-webkit-scrollbar-thumb {
           background-color: var(--fg-color);
           border-radius: 6px;
           border: 3px solid var(--bg-color);
        }
       }
        `
      document.head.appendChild(styleTag)
    }
  }

  /** Add documentation widget to the selected node */
  export const addDocumentation = (
    nodeData,
    nodeType,
    opts = { icon_size: 14, icon_margin: 4 },) => {

    opts = opts || {}
    const iconSize = opts.icon_size ? opts.icon_size : 14
    const iconMargin = opts.icon_margin ? opts.icon_margin : 4
    let docElement = null
    let contentWrapper = null
    //if no description in the node python code, don't do anything
    if (!nodeData.description) {
      return
    }

    const drawFg = nodeType.prototype.onDrawForeground
    nodeType.prototype.onDrawForeground = function (ctx) {
      const r = drawFg ? drawFg.apply(this, arguments) : undefined
      if (this.flags.collapsed) return r

      // icon position
      const x = this.size[0] - iconSize - iconMargin
      
      // create the popup
      if (this.show_doc && docElement === null) {
        docElement = document.createElement('div')
        contentWrapper = document.createElement('div');
        docElement.appendChild(contentWrapper);

        create_documentation_stylesheet()
        contentWrapper.classList.add('content-wrapper');
        docElement.classList.add('ollama-documentation-popup')
        
        // Simple plain text parsing (we don't import marked/DOMPurify to save space, relies on pre-formatted python text)
        let html_content = String(nodeData.description).replace(/\n/g, '<br/>');
        contentWrapper.innerHTML = `<strong>Node Help</strong><br/><br/>${html_content}`

        // resize handle
        const resizeHandle = document.createElement('div');
        resizeHandle.style.width = '0';
        resizeHandle.style.height = '0';
        resizeHandle.style.position = 'absolute';
        resizeHandle.style.bottom = '0';
        resizeHandle.style.right = '0';
        resizeHandle.style.cursor = 'se-resize';
        
        // Add pseudo-elements to create a triangle shape
        const borderColor = getComputedStyle(document.documentElement).getPropertyValue('--border-color').trim();
        resizeHandle.style.borderTop = '10px solid transparent';
        resizeHandle.style.borderLeft = '10px solid transparent';
        resizeHandle.style.borderBottom = `10px solid ${borderColor}`;
        resizeHandle.style.borderRight = `10px solid ${borderColor}`;

        docElement.appendChild(resizeHandle)
        let isResizing = false
        let startX, startY, startWidth, startHeight

        resizeHandle.addEventListener('mousedown', function (e) {
          e.preventDefault();
          e.stopPropagation();
          isResizing = true;
          startX = e.clientX;
          startY = e.clientY;
          startWidth = parseInt(document.defaultView.getComputedStyle(docElement).width, 10);
          startHeight = parseInt(document.defaultView.getComputedStyle(docElement).height, 10);
         },
         { signal: this.docCtrl?.signal },
         );

        // close button
        const closeButton = document.createElement('div');
        closeButton.textContent = '❌';
        closeButton.style.position = 'absolute';
        closeButton.style.top = '0';
        closeButton.style.right = '0';
        closeButton.style.cursor = 'pointer';
        closeButton.style.padding = '5px';
        closeButton.style.color = 'red';
        closeButton.style.fontSize = '12px';

        docElement.appendChild(closeButton)

        closeButton.addEventListener('mousedown', (e) => {
          e.stopPropagation();
          this.show_doc = !this.show_doc
          docElement.parentNode.removeChild(docElement)
          docElement = null
          if (contentWrapper) {
            contentWrapper.remove()
            contentWrapper = null
          }
         },
         { signal: this.docCtrl?.signal },
         );
         
        document.addEventListener('mousemove', function (e) {
          if (!isResizing) return;
          const scale = app.canvas.ds.scale;
          const newWidth = startWidth + (e.clientX - startX) / scale;
          const newHeight = startHeight + (e.clientY - startY) / scale;;
          docElement.style.width = `${newWidth}px`;
          docElement.style.height = `${newHeight}px`;
         },
         { signal: this.docCtrl?.signal },
         );

        document.addEventListener('mouseup', function () {
          isResizing = false
        },
        { signal: this.docCtrl?.signal },
        )

        document.body.appendChild(docElement)
      }
      // close the popup
      else if (!this.show_doc && docElement !== null) {
        docElement.parentNode.removeChild(docElement)
        docElement = null
      }
      // update position of the popup
      if (this.show_doc && docElement !== null) {
        const rect = ctx.canvas.getBoundingClientRect()
        const scaleX = rect.width / ctx.canvas.width
        const scaleY = rect.height / ctx.canvas.height

        const transform = new DOMMatrix()
        .scaleSelf(scaleX, scaleY)
        .multiplySelf(ctx.getTransform())
        .translateSelf(this.size[0] * scaleX * Math.max(1.0,window.devicePixelRatio) , 0)
        .translateSelf(10, -32)
        
        const scale = new DOMMatrix()
        .scaleSelf(transform.a, transform.d);
        const bcr = app.canvas.canvas.getBoundingClientRect()

        const styleObject = {
          transformOrigin: '0 0',
          transform: scale,
          left: `${transform.a + bcr.x + transform.e}px`,
          top: `${transform.d + bcr.y + transform.f}px`,
         };
        Object.assign(docElement.style, styleObject);
      }

      ctx.save()
      ctx.translate(x - 2, iconSize - 34)
      ctx.scale(iconSize / 32, iconSize / 32)
      ctx.strokeStyle = 'rgba(255,255,255,0.3)'
      ctx.lineCap = 'round'
      ctx.lineJoin = 'round'
      ctx.lineWidth = 2.4
      ctx.font = 'bold 36px monospace'
      ctx.fillStyle = 'orange';
      ctx.fillText('?', 0, 24)
      ctx.restore()
      return r
    }
    // handle clicking of the icon
    const mouseDown = nodeType.prototype.onMouseDown
    nodeType.prototype.onMouseDown = function (e, localPos, canvas) {
      const r = mouseDown ? mouseDown.apply(this, arguments) : undefined
      const iconX = this.size[0] - iconSize - iconMargin
      const iconY = iconSize - 34
      if (
        localPos[0] > iconX &&
        localPos[0] < iconX + iconSize &&
        localPos[1] > iconY &&
        localPos[1] < iconY + iconSize
      ) {
        if (this.show_doc === undefined) {
          this.show_doc = true
        } else {
          this.show_doc = !this.show_doc
        }
        if (this.show_doc) {
          this.docCtrl = new AbortController()
        } else {
          try { this.docCtrl.abort() } catch(e){}
        }
        return true;
      }
      return r;
    }
    const onRem = nodeType.prototype.onRemoved

    nodeType.prototype.onRemoved = function () {
      const r = onRem ? onRem.apply(this, []) : undefined
  
      if (docElement) {
        docElement.remove()
        docElement = null
      }
  
      if (contentWrapper) {
        contentWrapper.remove()
        contentWrapper = null
      }
      return r
    }
}
