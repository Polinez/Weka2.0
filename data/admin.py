from django.contrib import admin
from .models import Dataset, DatasetColumn


class DatasetColumnInline(admin.TabularInline):
    model = DatasetColumn
    extra = 0


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'problem_type', 'row_count', 'column_count', 'created_at')
    list_filter = ('problem_type', 'user')
    search_fields = ('name',)
    inlines = [DatasetColumnInline]
