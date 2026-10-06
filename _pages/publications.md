---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

You can also find my complete and up-to-date publication profile on [Google Scholar](https://scholar.google.com/citations?user=AE9hzbgAAAAJ&hl=en).

{% include base_path %}

{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}
