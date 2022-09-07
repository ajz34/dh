{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
   {% block classes %}
   {% if classes %}
   {% if objname not in ["options"] %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :toctree: .
      :template: custom-class-template.rst
      :nosignatures:
   {% for item in classes %}
      {{ item }}
   {% endfor %}
   {% else %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :template: custom-class-template.rst
      :nosignatures:
   {% for item in classes %}
      {{ item }}
   {% endfor %}

   {% endif %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Module Attributes

   .. autosummary::
      :nosignatures:
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}

   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :nosignatures:
   {% for item in functions %}
      ~{{ item }}
   {%- endfor %}

   {% endif %}
   {% endblock %}

   {% if objname in ["options"] %}
   {% if classes %}
   .. rubric:: Details of Classes
   {% for item in classes %}
   .. autoclass:: {{ item }}
     :members:
     :show-inheritance:
   {% endfor %}
   {% endif %}
   {% endif %}

   {% if attributes %}
   .. rubric:: Details of Attributes
   {% for item in attributes %}
   .. autodata:: {{ item }}
   {%- endfor %}
   {% endif %}

   {% if functions %}
   .. rubric:: Details of Functions
   {% for item in functions %}
   .. autofunction:: {{ item }}
   {%- endfor %}
   {% endif %}

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree: .
   :template: custom-module-template.rst
   :recursive:
   :nosignatures:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
