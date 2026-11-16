ARG PYTHON_IMAGE=python:3.12-slim@sha256:12316debbfb86890f6b2d825820612741ba527ad5a44e3902759141d572bfe8f
FROM ${PYTHON_IMAGE}
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY kubeai /app/kubeai
COPY config /app/config
RUN mkdir -p /app/results && chown -R 10001:10001 /app
USER 10001:10001
EXPOSE 8080
ENTRYPOINT ["python", "-m", "kubeai"]
CMD ["router", "--config", "config/docker.json", "--host", "0.0.0.0"]
