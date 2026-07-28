FROM rapidsai/base:26.06-cuda13-py3.12

WORKDIR /gpucputest

COPY compare_results.py /gpucputest/compare_results.py

ENTRYPOINT ["python", "/gpucputest/compare_results.py"]
