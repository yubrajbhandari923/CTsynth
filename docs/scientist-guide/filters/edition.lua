-- Edition filter. Same prose, two readings.
--   scientific: [text]{#clm-0001 .claim} becomes plain text.
--   researcher: the span links to its evidence entry, and each chapter gets its generated evidence panel appended.
--   both: HTML comments in the source are dropped from the output.
local edition = "scientific"

local function slug(path)
  local dir = quarto.project.directory or ""
  local rel = path
  if dir ~= "" and path:sub(1, #dir) == dir then rel = path:sub(#dir + 2) end
  rel = rel:gsub("%.qmd$", ""):gsub("/", "__")
  return rel
end

return {
  { Meta = function(m)
      if m.edition then edition = pandoc.utils.stringify(m.edition) end
    end },
  { RawBlock = function(el)  -- source comments may name internal records; they must not ship in either edition
      if el.format == "html" and el.text:match("^%s*<!%-%-") then return {} end
    end,
    RawInline = function(el)
      if el.format == "html" and el.text:match("^%s*<!%-%-") then return {} end
    end },
  { Span = function(el)
      if not el.classes:includes("claim") then return nil end
      if edition == "researcher" then
        local id = el.identifier
        local out = el.content
        if FORMAT:match("html") then  -- panels are per-file HTML pages; a merged PDF has neither them nor their anchors
          table.insert(out, pandoc.Superscript({ pandoc.Link(id:upper(), "#ev-" .. id) }))
        end
        return out
      end
      return el.content
    end },
  { Pandoc = function(doc)
      if edition ~= "researcher" then return nil end
      local input = quarto.doc.input_file
      if not input then return nil end
      local dir = quarto.project.directory or ""
      local f = io.open(dir .. "/_generated/evidence/" .. slug(input) .. ".md", "r")
      if not f then return nil end
      local text = f:read("*a")
      f:close()
      for _, b in ipairs(pandoc.read(text, "markdown").blocks) do doc.blocks:insert(b) end
      return doc
    end },
}
