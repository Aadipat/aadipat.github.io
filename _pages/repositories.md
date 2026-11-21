---
layout: page
permalink: /repositories/
title: repositories
description: Code behind my research and projects. Each repository links to its paper or write-up.
nav: true
nav_order: 4
---

{% for group in site.data.repositories.github_repos %}

## {{ group.section }}

<div class="repo-grid">
  {% for r in group.repos %}
    {% include repository/repo_card.liquid repo=r %}
  {% endfor %}
</div>
{% endfor %}

{% if site.data.repositories.github_users %}

## GitHub

<div class="repositories d-flex flex-wrap flex-md-row flex-column justify-content-between align-items-center">
  {% for user in site.data.repositories.github_users %}
    {% include repository/repo_user.liquid username=user %}
  {% endfor %}
</div>

All public repositories are on [github.com/Aadipat](https://github.com/Aadipat).
{% endif %}
